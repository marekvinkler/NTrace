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
#include "3d/Texture.hpp"

namespace FW
{
//------------------------------------------------------------------------

class TextureAtlas
{
private:
    struct Item
    {
        Texture             texture;
        S32                 border;
        bool                wrap;
        Vec2i               size;
        Vec2i               pos;
    };

public:
                            TextureAtlas    (const ImageFormat& format = ImageFormat::ABGR_8888);
                            ~TextureAtlas   (void);

    void                    clear           (void);
    bool                    addTexture      (const Texture& tex, int border = 1, bool wrap = true);

    Vec2i                   getAtlasSize    (void)  { validate(); return m_atlasSize; }
    Vec2i                   getTexturePos   (const Texture& tex);
	Vec2i					getTextureSize	(const Texture& tex);
	Vec2f					getTexturePosF	(const Texture& tex);
	Vec2f					getTextureSizeF	(const Texture& tex);
    const Texture&          getAtlasTexture (void)  { validate(); return m_atlasTexture; }

private:
                            TextureAtlas    (const TextureAtlas&); // forbidden
    TextureAtlas&           operator=       (const TextureAtlas&); // forbidden

private:
    void                    validate        (void);
    void                    layoutItems     (void);
    void                    createAtlas     (void);

private:
    ImageFormat             m_format;
    Array<Item>             m_items;
    Hash<const Image*, S32> m_itemHash;

    Vec2i                   m_atlasSize;
    Texture                 m_atlasTexture;
};

//------------------------------------------------------------------------
}
