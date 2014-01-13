/*
*  Copyright (c) 2013, Vilem Otte
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of the <organization> nor the
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

#ifndef __ACCELERATION__H__
#define __ACCELERATION__H__

#include "Scene.hpp"
#include "bvh\Platform.hpp"

namespace FW
{
	/**
	  * Acceleration structure base class
	  * @name AccelerationStructure
	  * @desc Base class for creating other acceleration structures. 
	  **/
	class AccelerationStructure
	{
	protected:
		//////////////////////////
		/** @section Variables **/
		Scene*				m_scene;		//< Pointer to scene, on which acc. structure is built
		Platform			m_platform;		//< Platform information

	public:
		/////////////////////////////
		/** @section Constructors **/

		/** 
		  * Default constructor
		  * @name AccelerationStructure
		  * @param Scene* - scene on which acc. structure is to be built
		  * @param const Platform& - const. ref. to platform info
		  * @return None
		  **/
		AccelerationStructure(Scene* scene, const Platform& platform);

		////////////////////////////
		/** @section Destructors **/

		/**
		  * Default destructor (virtual)
		  * @name ~AccelerationStructure
		  * @param None
		  * @return None
		  **/
		virtual ~AccelerationStructure();

		////////////////////////
		/** @section Methods **/

		/**
		  * Getter for scene
		  * @name getScene
		  * @param None
		  * @return Scene*, pointing to scene on which Acc. structure was built
		  **/
		Scene*              getScene(void) const				{ return m_scene; }

		/**
		  * Getter for platform
		  * @name getPlatform
		  * @param None
		  * @return const ref. to platform
		  **/
		const Platform&     getPlatform(void) const				{ return m_platform; }
	};
}

#endif