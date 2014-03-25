
/**
* \file
* \brief Definitions for Cuda KDTree
*/

#pragma once
#include "gpu/Buffer.hpp"
#include "io/Stream.hpp"
#include "kdtree/KDTree.hpp"
#include "CudaAS.hpp"
#include "kernels/CudaTracerKernels.hpp"

namespace FW
{
/**
 *  \brief Cuda friendly KDTree representation.
 *  \details Converts KDTree data to buffers optimal for traversal and storage.
 */
class CudaKDTree : public CudaAS
{
public:
	BVHLayout   getLayout           (void) const            { return BVHLayout::BVHLayout_Compact; }

	/**
	*  \brief				Constructor.
	*  \pram[in] kdtree		KDTree to convert.
	*/
	explicit	CudaKDTree			(const KDTree& kdtree);

	/**
	*  \brief				Copy constructor.
	*  \pram[in] kdtree		KDTree to copy.
	*/
				CudaKDTree			(CudaKDTree& other)			{ operator=(other); }

	/**
	*  \brief				Constructs CudaKDTree from an input stream.
	*  \pram[in] in			Source input stream.
	*/
	explicit	CudaKDTree			(InputStream& in);

	/**
	*  \brief				Destructor.
	*/
				~CudaKDTree			(void);

	/**
	*  \brief				Returns node buffer.
	*  \return				Buffer containing nodes.
	*/
	Buffer&		getNodeBuffer		(void)						{ return m_nodes; }

	/**
	*  \brief				Returns triangle index buffer.
	*  \return				Buffer containing indexes which point to the scene's triangle buffer.
	*/
	Buffer& 	getTriIndexBuffer	(void)						{ return m_triIndex; }

	/**
	*  \brief				Returns buffer of woopified triangles.
	*  \return				Buffer containing wooopified triangles. These triangles are
	*						in the same order as they are in the scene's triangle buffer.
	*/
	Buffer&		getTriWoopBuffer	(void)						{ return m_triWoop; }

	/**
	*  \brief				Writes CudaKDTree to a given output stream.
	*  \param[in] out		Output stream to write CudaKDTree to.
	*/
	void		serialize			(OutputStream& out);

	/**
	*  \brief				Returns bounding box of the CudaKDTree's source scene.
	*  \return				Bounding box of the CudaKDTree's source scene.
	*/
	const AABB& getBBox				(void) const					{ return m_bbox; }

private:
	/**
	*  \brief				Internal structure used in conversion.
	*/
	struct StackEntry
	{
		const KDTreeNode*	node; //!<KDTree node this stack entry is related to. 
		S32					idx; //!<Index in either node buffer or triangle index buffer. Depends on whether the node is a leaf node or an inner node.

		/**
		*  \brief			Constructor.
		*  \param[in] n		KDTree node this stack entry should be related to.
		*  \param[in] i		Index of this entry.
		*/
		StackEntry(const KDTreeNode* n = NULL, int i = 0) : node(n), idx(i) {}

		/**
		*  \brief			Encodes index.
		*  \return          Index of this entry's node. Positive if the node is an inner node, negative otherwise.
		*/
		int encodeIdx(void) const { return (node->isLeaf()) ? ~idx : idx; }
	};
	
private:

	/**
	*  \brief				Converts KDTree's nodes and triangle indexes to buffers.
	*  \param[in] kdtree	KDTree to convert.
	*/
	void				createNodeTriIdx	(const KDTree& kdtree);

	/**
	*  \brief				Converts KDTree's source scene triangles to their woopified version.
	*  \param[in] kdtree	KDTree whose source scene triangles should be converted.
	*/
	void				createWoopTri		(const KDTree& kdtree);

	Buffer				m_nodes;		//!< Buffer holding nodes.
	Buffer				m_triIndex;		//!< Buffer holding triangle indexes pointing to the scene's triangle buffer.
	Buffer				m_triWoop;		//!< Buffer holding whoopified version of scene's triangles.

	AABB				m_bbox;			//!< Bounding box of the tree.
};

}