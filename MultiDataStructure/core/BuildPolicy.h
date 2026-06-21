#pragma once

#include "../stdafx.h"

struct BuildPolicy
{
	size_t maxDepth = 0;
	size_t leafCapacity = 1;
	size_t minPrimitivesToSplit = 2;
	bool collapseSingleChild = true;
	bool removeEmptyNodes = true;
	bool allowOverlapDuplication = true;
	bool enableLeafMicroIndexes = false;
	size_t leafMicroIndexThreshold = 512;
};
