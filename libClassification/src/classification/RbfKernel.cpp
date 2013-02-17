/*
 * RbfKernel.cpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#include "classification/RbfKernel.hpp"

namespace classification {

RbfKernel::RbfKernel(void)
{
}

RbfKernel::RbfKernel(double gamma) : gamma(gamma)
{
}

RbfKernel::~RbfKernel(void)
{
}

} /* namespace classification */
