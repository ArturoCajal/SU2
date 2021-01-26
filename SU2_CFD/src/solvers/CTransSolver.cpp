/*!
 * \file CTransSolver.cpp
 * \brief Main subrotuines of CTransSolver class
 * \author A. Cajal.
 * \version 7.1.0 "Blackbird"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2020, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "../../include/solvers/CTransSolver.hpp"
#include "../../../Common/include/parallelization/omp_structure.hpp"


CTransSolver::CTransSolver(void) : CSolver() { }

CTransSolver::CTransSolver(CGeometry* geometry, CConfig *config) : CSolver() {

  Gamma = config->GetGamma();
  Gamma_Minus_One = Gamma - 1.0;

  nMarker = config->GetnMarker_All();

  /*--- Store the number of vertices on each marker for deallocation later ---*/
  nVertex = new unsigned long[nMarker];
  for (unsigned long iMarker = 0; iMarker < nMarker; iMarker++)
    nVertex[iMarker] = geometry->nVertex[iMarker];

  /* A grid is defined as dynamic if there's rigid grid movement or grid deformation AND the problem is time domain */
  dynamic_grid = config->GetDynamic_Grid();

#ifdef HAVE_OMP
  /*--- Get the edge coloring, see notes in CEulerSolver's constructor. ---*/
  su2double parallelEff = 1.0;
  const auto& coloring = geometry->GetEdgeColoring(&parallelEff);

  ReducerStrategy = parallelEff < COLORING_EFF_THRESH;

  if (ReducerStrategy && (coloring.getOuterSize()>1))
    geometry->SetNaturalEdgeColoring();

  if (!coloring.empty()) {
    auto groupSize = ReducerStrategy? 1ul : geometry->GetEdgeColorGroupSize();
    auto nColor = coloring.getOuterSize();
    EdgeColoring.reserve(nColor);

    for(auto iColor = 0ul; iColor < nColor; ++iColor)
      EdgeColoring.emplace_back(coloring.innerIdx(iColor), coloring.getNumNonZeros(iColor), groupSize);
  }

  nPoint = geometry->GetnPoint();
  omp_chunk_size = computeStaticChunkSize(nPoint, omp_get_max_threads(), OMP_MAX_SIZE);
#else
  EdgeColoring[0] = DummyGridColor<>(geometry->GetnEdge());
#endif
}

CTransSolver::~CTransSolver(void) {

  if (Inlet_TransVars != nullptr) {
    for (unsigned short iMarker = 0; iMarker < nMarker; iMarker++) {
      if (Inlet_TransVars[iMarker] != nullptr) {
        for (unsigned long iVertex = 0; iVertex < nVertex[iMarker]; iVertex++) {
          delete [] Inlet_TransVars[iMarker][iVertex];
        }
        delete [] Inlet_TransVars[iMarker];
      }
    }
    delete [] Inlet_TransVars;
  }

  delete nodes;
}

void CTransSolver::Upwind_Residual(CGeometry *geometry, CSolver **solver_container,
                                  CNumerics **numerics_container, CConfig *config, unsigned short iMesh) {

  const bool muscl = config->GetMUSCL_Turb();
  const bool limiter = (config->GetKind_SlopeLimit_Turb() != NO_LIMITER);

  /*--- Only reconstruct flow variables if MUSCL is on for flow (requires upwind) and turbulence. ---*/
  const bool musclFlow = config->GetMUSCL_Flow() && muscl &&
                        (config->GetKind_ConvNumScheme_Flow() == SPACE_UPWIND);
  /*--- Only consider flow limiters for cell-based limiters, edge-based would need to be recomputed. ---*/
  const bool limiterFlow = (config->GetKind_SlopeLimit_Flow() != NO_LIMITER) &&
                           (config->GetKind_SlopeLimit_Flow() != VAN_ALBADA_EDGE);

  CVariable* flowNodes = solver_container[FLOW_SOL]->GetNodes();

  /*--- Pick one numerics object per thread. ---*/
  CNumerics* numerics = numerics_container[CONV_TERM + omp_get_thread_num()*MAX_TERMS];

  /*--- Static arrays of MUSCL-reconstructed flow primitives and transition variables (thread safety). ---*/
  su2double solution_i[MAXNVAR] = {0.0}, flowPrimVar_i[MAXNVARFLOW] = {0.0};
  su2double solution_j[MAXNVAR] = {0.0}, flowPrimVar_j[MAXNVARFLOW] = {0.0};

  /*--- Loop over edge colors. ---*/
  for (auto color : EdgeColoring)
  {
  /*--- Chunk size is at least OMP_MIN_SIZE and a multiple of the color group size. ---*/
  SU2_OMP_FOR_DYN(nextMultiple(OMP_MIN_SIZE, color.groupSize))
  for(auto k = 0ul; k < color.size; ++k) {

    auto iEdge = color.indices[k];

    unsigned short iDim, iVar;

    /*--- Points in edge and normal vectors ---*/

    auto iPoint = geometry->edges->GetNode(iEdge,0);
    auto jPoint = geometry->edges->GetNode(iEdge,1);

    numerics->SetNormal(geometry->edges->GetNormal(iEdge));

    /*--- Primitive variables w/o reconstruction ---*/

    const auto V_i = flowNodes->GetPrimitive(iPoint);
    const auto V_j = flowNodes->GetPrimitive(jPoint);
    numerics->SetPrimitive(V_i, V_j);

    /*--- Transition variables w/o reconstruction ---*/

    const auto Trans_i = nodes->GetSolution(iPoint);
    const auto Trans_j = nodes->GetSolution(jPoint);
    numerics->SetTransVar(Trans_i, Trans_j);

    /*--- Grid Movement ---*/

    if (dynamic_grid)
      numerics->SetGridVel(geometry->nodes->GetGridVel(iPoint),
                           geometry->nodes->GetGridVel(jPoint));

    if (muscl || musclFlow) {
      const su2double *Limiter_i = nullptr, *Limiter_j = nullptr;

      const auto Coord_i = geometry->nodes->GetCoord(iPoint);
      const auto Coord_j = geometry->nodes->GetCoord(jPoint);

      su2double Vector_ij[MAXNDIM] = {0.0};
      for (iDim = 0; iDim < nDim; iDim++) {
        Vector_ij[iDim] = 0.5*(Coord_j[iDim] - Coord_i[iDim]);
      }

      if (musclFlow) {
        /*--- Reconstruct mean flow primitive variables. ---*/

        auto Gradient_i = flowNodes->GetGradient_Reconstruction(iPoint);
        auto Gradient_j = flowNodes->GetGradient_Reconstruction(jPoint);

        if (limiterFlow) {
          Limiter_i = flowNodes->GetLimiter_Primitive(iPoint);
          Limiter_j = flowNodes->GetLimiter_Primitive(jPoint);
        }

        for (iVar = 0; iVar < solver_container[FLOW_SOL]->GetnPrimVarGrad(); iVar++) {
          su2double Project_Grad_i = 0.0, Project_Grad_j = 0.0;
          for (iDim = 0; iDim < nDim; iDim++) {
            Project_Grad_i += Vector_ij[iDim]*Gradient_i[iVar][iDim];
            Project_Grad_j -= Vector_ij[iDim]*Gradient_j[iVar][iDim];
          }
          if (limiterFlow) {
            Project_Grad_i *= Limiter_i[iVar];
            Project_Grad_j *= Limiter_j[iVar];
          }
          flowPrimVar_i[iVar] = V_i[iVar] + Project_Grad_i;
          flowPrimVar_j[iVar] = V_j[iVar] + Project_Grad_j;
        }

        numerics->SetPrimitive(flowPrimVar_i, flowPrimVar_j);
      }

      if (muscl) {
        /*--- Reconstruct transition variables. ---*/

        auto Gradient_i = nodes->GetGradient_Reconstruction(iPoint);
        auto Gradient_j = nodes->GetGradient_Reconstruction(jPoint);

        if (limiter) {
          Limiter_i = nodes->GetLimiter(iPoint);
          Limiter_j = nodes->GetLimiter(jPoint);
        }

        for (iVar = 0; iVar < nVar; iVar++) {
          su2double Project_Grad_i = 0.0, Project_Grad_j = 0.0;
          for (iDim = 0; iDim < nDim; iDim++) {
            Project_Grad_i += Vector_ij[iDim]*Gradient_i[iVar][iDim];
            Project_Grad_j -= Vector_ij[iDim]*Gradient_j[iVar][iDim];
          }
          if (limiter) {
            Project_Grad_i *= Limiter_i[iVar];
            Project_Grad_j *= Limiter_j[iVar];
          }
          solution_i[iVar] = Trans_i[iVar] + Project_Grad_i;
          solution_j[iVar] = Trans_j[iVar] + Project_Grad_j;
        }

        numerics->SetTransVar(solution_i, solution_j);
      }
    }

    /*--- Update convective residual value ---*/

    auto residual = numerics->ComputeResidual(config);

    if (ReducerStrategy) {
      EdgeFluxes.SetBlock(iEdge, residual);
      Jacobian.SetBlocks(iEdge, residual.jacobian_i, residual.jacobian_j);
    }
    else {
      LinSysRes.AddBlock(iPoint, residual);
      LinSysRes.SubtractBlock(jPoint, residual);
      Jacobian.UpdateBlocks(iEdge, iPoint, jPoint, residual.jacobian_i, residual.jacobian_j);
    }

    /*--- Viscous contribution. ---*/

    Viscous_Residual(iEdge, geometry, solver_container,
                     numerics_container[VISC_TERM + omp_get_thread_num()*MAX_TERMS], config);
  }
  } // end color loop

  if (ReducerStrategy) {
    SumEdgeFluxes(geometry);
    Jacobian.SetDiagonalAsColumnSum();
  }
}

void CTransSolver::Viscous_Residual(unsigned long iEdge, CGeometry *geometry, CSolver **solver_container,
                                   CNumerics *numerics, CConfig *config) {

  CVariable* flowNodes = solver_container[FLOW_SOL]->GetNodes();

  /*--- Points in edge ---*/

  auto iPoint = geometry->edges->GetNode(iEdge,0);
  auto jPoint = geometry->edges->GetNode(iEdge,1);

  /*--- Points coordinates, and normal vector ---*/

  numerics->SetCoord(geometry->nodes->GetCoord(iPoint),
                     geometry->nodes->GetCoord(jPoint));
  numerics->SetNormal(geometry->edges->GetNormal(iEdge));

  /*--- Conservative variables w/o reconstruction ---*/

  numerics->SetPrimitive(flowNodes->GetPrimitive(iPoint),
                         flowNodes->GetPrimitive(jPoint));

  /*--- Transitional variables w/o reconstruction, and its gradients ---*/

  numerics->SetTransVar(nodes->GetSolution(iPoint),
                       nodes->GetSolution(jPoint));
  numerics->SetTransVarGradient(nodes->GetGradient(iPoint),
                               nodes->GetGradient(jPoint));

  /*--- Compute residual, and Jacobians ---*/

  auto residual = numerics->ComputeResidual(config);

  if (ReducerStrategy) {
    EdgeFluxes.SubtractBlock(iEdge, residual);
    Jacobian.UpdateBlocksSub(iEdge, residual.jacobian_i, residual.jacobian_j);
  }
  else {
    LinSysRes.SubtractBlock(iPoint, residual);
    LinSysRes.AddBlock(jPoint, residual);
    Jacobian.UpdateBlocksSub(iEdge, iPoint, jPoint, residual.jacobian_i, residual.jacobian_j);
  }
}

void CTransSolver::SumEdgeFluxes(CGeometry* geometry) {

  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iPoint = 0; iPoint < nPoint; ++iPoint) {

    LinSysRes.SetBlock_Zero(iPoint);

    for (auto iEdge : geometry->nodes->GetEdges(iPoint)) {
      if (iPoint == geometry->edges->GetNode(iEdge,0))
        LinSysRes.AddBlock(iPoint, EdgeFluxes.GetBlock(iEdge));
      else
        LinSysRes.SubtractBlock(iPoint, EdgeFluxes.GetBlock(iEdge));
    }
  }

}

void CTransSolver::BC_Sym_Plane(CGeometry      *geometry,
                               CSolver        **solver_container,
                               CNumerics      *conv_numerics,
                               CNumerics      *visc_numerics,
                               CConfig        *config,
                               unsigned short val_marker) {

  /*--- Convective and viscous fluxes across symmetry plane are equal to zero. ---*/

}

void CTransSolver::BC_Euler_Wall(CGeometry      *geometry,
                                CSolver        **solver_container,
                                CNumerics      *conv_numerics,
                                CNumerics      *visc_numerics,
                                CConfig        *config,
                                unsigned short val_marker) {

  /*--- Convective fluxes across euler wall are equal to zero. ---*/

}

void CTransSolver::BC_Riemann(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  string Marker_Tag         = config->GetMarker_All_TagBound(val_marker);

  switch(config->GetKind_Data_Riemann(Marker_Tag))
  {
  case TOTAL_CONDITIONS_PT: case STATIC_SUPERSONIC_INFLOW_PT: case STATIC_SUPERSONIC_INFLOW_PD: case DENSITY_VELOCITY:
    BC_Inlet(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
    break;
  case STATIC_PRESSURE:
    BC_Outlet(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
    break;
  }
}

void CTransSolver::BC_TurboRiemann(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  string Marker_Tag         = config->GetMarker_All_TagBound(val_marker);

  switch(config->GetKind_Data_Riemann(Marker_Tag))
  {
  case TOTAL_CONDITIONS_PT: case STATIC_SUPERSONIC_INFLOW_PT: case STATIC_SUPERSONIC_INFLOW_PD: case DENSITY_VELOCITY:
    BC_Inlet_Turbo(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
    break;
  case STATIC_PRESSURE:
    BC_Outlet(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
    break;
  }
}


void CTransSolver::BC_Giles(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  string Marker_Tag         = config->GetMarker_All_TagBound(val_marker);

  switch(config->GetKind_Data_Giles(Marker_Tag))
  {
  case TOTAL_CONDITIONS_PT:case TOTAL_CONDITIONS_PT_1D: case DENSITY_VELOCITY:
    BC_Inlet_Turbo(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
    break;
  case MIXING_IN:
    if (config->GetBoolTurbMixingPlane()){
      BC_Inlet_MixingPlane(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
    }
    else{
      BC_Inlet_Turbo(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
    }
    break;

  case STATIC_PRESSURE: case MIXING_OUT: case STATIC_PRESSURE_1D: case RADIAL_EQUILIBRIUM:
    BC_Outlet(geometry, solver_container, conv_numerics, visc_numerics, config, val_marker);
    break;
  }
}

void CTransSolver::BC_Periodic(CGeometry *geometry, CSolver **solver_container,
                                  CNumerics *numerics, CConfig *config) {

  /*--- Complete residuals for periodic boundary conditions. We loop over
   the periodic BCs in matching pairs so that, in the event that there are
   adjacent periodic markers, the repeated points will have their residuals
   accumulated corectly during the communications. For implicit calculations
   the Jacobians and linear system are also correctly adjusted here. ---*/

  for (unsigned short iPeriodic = 1; iPeriodic <= config->GetnMarker_Periodic()/2; iPeriodic++) {
    InitiatePeriodicComms(geometry, config, iPeriodic, PERIODIC_RESIDUAL);
    CompletePeriodicComms(geometry, config, iPeriodic, PERIODIC_RESIDUAL);
  }

}

void CTransSolver::BC_Fluid_Interface(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics,
                                     CNumerics *visc_numerics, CConfig *config) {

  /*--- Not implemented yet. See CTurbSolver::BC_Fluid_Interface. ---*/

}

void CTransSolver::ImplicitEuler_Iteration(CGeometry *geometry, CSolver **solver_container, CConfig *config) {

  const bool adjoint = config->GetContinuous_Adjoint() || (config->GetDiscrete_Adjoint() && config->GetFrozen_Visc_Disc());
  const bool compressible = (config->GetKind_Regime() == COMPRESSIBLE);

  CVariable* flowNodes = solver_container[FLOW_SOL]->GetNodes();

  /*--- Set shared residual variables to 0 and declare
   *    local ones for current thread to work on. ---*/

  SU2_OMP_MASTER
  for (unsigned short iVar = 0; iVar < nVar; iVar++) {
    SetRes_RMS(iVar, 0.0);
    SetRes_Max(iVar, 0.0, 0);
  }
  SU2_OMP_BARRIER

  su2double resMax[MAXNVAR] = {0.0}, resRMS[MAXNVAR] = {0.0};
  const su2double* coordMax[MAXNVAR] = {nullptr};
  unsigned long idxMax[MAXNVAR] = {0};

  /*--- Build implicit system ---*/

  SU2_OMP(for schedule(static,omp_chunk_size) nowait)
  for (unsigned long iPoint = 0; iPoint < nPointDomain; iPoint++) {

    /*--- Read the volume ---*/

    su2double Vol = (geometry->nodes->GetVolume(iPoint) + geometry->nodes->GetPeriodicVolume(iPoint));

    /*--- Modify matrix diagonal to assure diagonal dominance ---*/

    su2double Delta = Vol / ((nodes->GetLocalCFL(iPoint)/flowNodes->GetLocalCFL(iPoint))*flowNodes->GetDelta_Time(iPoint));
    Jacobian.AddVal2Diag(iPoint, Delta);

    /*--- Right hand side of the system (-Residual) and initial guess (x = 0) ---*/

    for (unsigned short iVar = 0; iVar < nVar; iVar++) {
      unsigned long total_index = iPoint*nVar + iVar;
      LinSysRes[total_index] = -LinSysRes[total_index];
      LinSysSol[total_index] = 0.0;

      su2double Res = fabs(LinSysRes[total_index]);
      resRMS[iVar] += Res*Res;
      if (Res > resMax[iVar]) {
        resMax[iVar] = Res;
        idxMax[iVar] = iPoint;
        coordMax[iVar] = geometry->nodes->GetCoord(iPoint);
      }
    }
  }
  SU2_OMP_CRITICAL
  for (unsigned short iVar = 0; iVar < nVar; iVar++) {
    AddRes_RMS(iVar, resRMS[iVar]);
    AddRes_Max(iVar, resMax[iVar], geometry->nodes->GetGlobalIndex(idxMax[iVar]), coordMax[iVar]);
  }

  /*--- Initialize residual and solution at the ghost points ---*/

  SU2_OMP(sections)
  {
    SU2_OMP(section)
    for (unsigned long iPoint = nPointDomain; iPoint < nPoint; iPoint++)
      LinSysRes.SetBlock_Zero(iPoint);

    SU2_OMP(section)
    for (unsigned long iPoint = nPointDomain; iPoint < nPoint; iPoint++)
      LinSysSol.SetBlock_Zero(iPoint);
  }

  /*--- Solve or smooth the linear system ---*/

  auto iter = System.Solve(Jacobian, LinSysRes, LinSysSol, geometry, config);
  SU2_OMP_MASTER
  {
    SetIterLinSolver(iter);
    SetResLinSolver(System.GetResidual());
  }
  SU2_OMP_BARRIER


  ComputeUnderRelaxationFactor(solver_container, config);

  /*--- Update solution (system written in terms of increments) ---*/

  if (!adjoint) {

    /*--- Update the transition solution. Only LM variants are clipped. ---*/

    switch (config->GetKind_Trans_Model()) {

      case LKE:

        SU2_OMP_FOR_STAT(omp_chunk_size)
        for (unsigned long iPoint = 0; iPoint < nPointDomain; iPoint++) {
            for (unsigned short iVar = 0; iVar < nVar; iVar++) {
                nodes->AddSolution(iPoint, iVar, nodes->GetUnderRelaxation(iPoint)*LinSysSol[iPoint]);
            }
        }
        break;

      case LM:

        SU2_OMP_FOR_STAT(omp_chunk_size)
        for (unsigned long iPoint = 0; iPoint < nPointDomain; iPoint++) {

          su2double density = flowNodes->GetDensity(iPoint);
          su2double density_old = density;

          if (compressible)
            density_old = flowNodes->GetSolution_Old(iPoint,0);

          for (unsigned short iVar = 0; iVar < nVar; iVar++) {
            nodes->AddConservativeSolution(iPoint, iVar,
                      nodes->GetUnderRelaxation(iPoint)*LinSysSol(iPoint,iVar),
                      density, density_old, lowerlimit[iVar], upperlimit[iVar]);
          }
        }
        break;

    }
  }

  for (unsigned short iPeriodic = 1; iPeriodic <= config->GetnMarker_Periodic()/2; iPeriodic++) {
    InitiatePeriodicComms(geometry, config, iPeriodic, PERIODIC_IMPLICIT);
    CompletePeriodicComms(geometry, config, iPeriodic, PERIODIC_IMPLICIT);
  }

  /*--- MPI solution ---*/

  InitiateComms(geometry, config, SOLUTION_EDDY);
  CompleteComms(geometry, config, SOLUTION_EDDY);

  /*--- Compute the root mean square residual ---*/
  SU2_OMP_MASTER
  SetResidual_RMS(geometry, config);
  SU2_OMP_BARRIER

}

void CTransSolver::ComputeUnderRelaxationFactor(CSolver **solver_container, const CConfig *config) {

  /* Loop over the solution update given by relaxing the linear
   system for this nonlinear iteration. */

  su2double localUnderRelaxation =  1.00;

  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iPoint = 0; iPoint < nPointDomain; iPoint++) {

    localUnderRelaxation = 1.0;

    /* Threshold the relaxation factor in the event that there is
     a very small value. This helps avoid catastrophic crashes due
     to non-realizable states by canceling the update. */

    if (localUnderRelaxation < 1e-10) localUnderRelaxation = 0.0;

    /* Store the under-relaxation factor for this point. */

    nodes->SetUnderRelaxation(iPoint, localUnderRelaxation);

  }

}

void CTransSolver::SetResidual_DualTime(CGeometry *geometry, CSolver **solver_container, CConfig *config,
                                       unsigned short iRKStep, unsigned short iMesh, unsigned short RunTime_EqSystem) {

  /*--- Not implemented yet. See CTurbSolver::SetResidual_DualTime. ---*/
}


void CTransSolver::LoadRestart(CGeometry **geometry, CSolver ***solver, CConfig *config, int val_iter, bool val_update_geo) {

  /*--- Not implemented yet. See CTurbSolver::LoadRestart. ---*/
}
