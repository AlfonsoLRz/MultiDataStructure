#pragma once

#include "../stdafx.h"

struct BuildPolicy
{
	size_t	_maxDepth = 0;
	size_t	_leafCapacity = 1;
	size_t	_minPrimitivesToSplit = 2;
	bool	_collapseSingleChild = true;
	bool	_removeEmptyNodes = true;
	bool	_allowOverlapDuplication = true;
	bool	_enableLeafMicroIndexes = false;
	size_t	_leafMicroIndexThreshold = 512;
};
