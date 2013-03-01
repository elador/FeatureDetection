/*
 * LoggerFactory.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LOGGERFACTORY_HPP_
#define LOGGERFACTORY_HPP_

namespace logging {

#define LoggerFacto LoggerFactory::Instance()

class LoggerFactory
{
private:
	LoggerFactory();
	~LoggerFactory();
	LoggerFactory(const LoggerFactory &);
	LoggerFactory& operator=(const LoggerFactory &);

public:
	static LoggerFactory* Instance();

};

} /* namespace logging */
#endif /* LOGGERFACTORY_HPP_ */
