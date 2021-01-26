/*!
 * \file CTransVariable.hpp
 * \brief Base class for defining the variables of the transition model.
 * \author A. Cajal
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

#pragma once

#include "CVariable.hpp"

/*!
 * \class CTransVariable
 * \brief Base class for defining the variables of the transition model.
 * \ingroup Transition_Model
 * \author A. Cajal.
 */
class CTransVariable : public CVariable {
protected:
  MatrixType HB_Source;   /*!< \brief Harmonic Balance source term. */

  CVectorOfMatrix& Gradient_Reconstruction;  /*!< \brief Reference to the gradient of the primitive variables for MUSCL reconstruction for the convective term */
  CVectorOfMatrix Gradient_Aux;              /*!< \brief Auxiliary structure to store a second gradient for reconstruction, if required. */

public:
  /*!
   * \brief Constructor of the class.
   * \param[in] npoint - Number of points/nodes/vertices in the domain.
   * \param[in] ndim - Number of dimensions of the problem.
   * \param[in] nvar - Number of variables of the problem.
   * \param[in] config - Definition of the particular problem.
   */
  CTransVariable(unsigned long npoint, unsigned long ndim, unsigned long nvar, CConfig *config);

  /*!
   * \brief Destructor of the class.
   */
  ~CTransVariable() override = default;

  /*!
   * \brief Get the value of the reconstruction variables gradient at a node.
   * \param[in] iPoint - Index of the current node.
   * \param[in] iVar   - Index of the variable.
   * \param[in] iDim   - Index of the dimension.
   * \return Value of the reconstruction variables gradient at a node.
   */
  inline su2double GetGradient_Reconstruction(unsigned long iPoint, unsigned long iVar, unsigned long iDim) const final {
    return Gradient_Reconstruction(iPoint,iVar,iDim);
  }

  /*!
   * \brief Get the value of the reconstruction variables gradient at a node.
   * \param[in] iPoint - Index of the current node.
   * \param[in] iVar   - Index of the variable.
   * \param[in] iDim   - Index of the dimension.
   * \param[in] value  - Value of the reconstruction gradient component.
   */
  inline void SetGradient_Reconstruction(unsigned long iPoint, unsigned long iVar, unsigned long iDim, su2double value) final {
    Gradient_Reconstruction(iPoint,iVar,iDim) = value;
  }

  /*!
   * \brief Get the array of the reconstruction variables gradient at a node.
   * \param[in] iPoint - Index of the current node.
   * \return Array of the reconstruction variables gradient at a node.
   */
  inline su2double **GetGradient_Reconstruction(unsigned long iPoint) final { return Gradient_Reconstruction[iPoint]; }

  /*!
   * \brief Get the reconstruction gradient for primitive variable at all points.
   * \return Reference to variable reconstruction gradient.
   */
  inline CVectorOfMatrix& GetGradient_Reconstruction(void) final { return Gradient_Reconstruction; }

};
