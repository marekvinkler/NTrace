/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include "base/DLLImports.hpp"

namespace FW
{
//------------------------------------------------------------------------

class Timer
{
public:
    explicit inline     Timer           (bool started = false)  : m_startTicks((started) ? queryTicks() : -1), m_totalTicks(0) {}
    inline              Timer           (const Timer& other)    { operator=(other); }
    inline              ~Timer          (void)                  {}

    inline void         start           (void)                  { m_startTicks = queryTicks(); }
    inline void         unstart         (void)                  { m_startTicks = -1; }
    inline F32          getElapsed      (void)                  { return ticksToSecs(getElapsedTicks()); }

    inline F32          end             (void);                 // return elapsed, total += elapsed, restart
    inline F32          getTotal        (void) const            { return ticksToSecs(m_totalTicks); }
    inline void         clearTotal      (void)                  { m_totalTicks = 0; }

    inline Timer&       operator=       (const Timer& other);

    static void         staticInit      (void);
    static inline S64   queryTicks      (void);
    static inline F32   ticksToSecs     (S64 ticks);

private:
    inline S64          getElapsedTicks (void);                 // return time since start, start if unstarted

private:
    static F64      s_ticksToSecsCoef;
    static S64      s_prevTicks;

    S64             m_startTicks;
    S64             m_totalTicks;
};

//------------------------------------------------------------------------

F32 Timer::end(void)
{
    S64 elapsed = getElapsedTicks();
    m_startTicks += elapsed;
    m_totalTicks += elapsed;
    return ticksToSecs(elapsed);
}

//------------------------------------------------------------------------

Timer& Timer::operator=(const Timer& other)
{
    m_startTicks = other.m_startTicks;
    m_totalTicks = other.m_totalTicks;
    return *this;
}

//------------------------------------------------------------------------

S64 Timer::queryTicks(void)
{
    LARGE_INTEGER ticks;
    QueryPerformanceCounter(&ticks);
    ticks.QuadPart = max(s_prevTicks, ticks.QuadPart);
    s_prevTicks = ticks.QuadPart; // increasing little endian => thread-safe
    return ticks.QuadPart;
}

//------------------------------------------------------------------------

F32 Timer::ticksToSecs(S64 ticks)
{
    if (s_ticksToSecsCoef == -1.0)
        staticInit();
    return (F32)((F64)ticks * s_ticksToSecsCoef);
}

//------------------------------------------------------------------------

S64 Timer::getElapsedTicks(void)
{
    S64 curr = queryTicks();
    if (m_startTicks == -1)
        m_startTicks = curr;
    return curr - m_startTicks;
}

//------------------------------------------------------------------------
}
