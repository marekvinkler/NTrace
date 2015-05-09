#pragma once

#include "persistentds/PersistentHelper.hpp"
#include "persistentds/CudaPool.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace std;

namespace FW
{

void allocateSnapshots(CudaModule* module, Buffer &snapData)
{
	// Prepare snapshot memory
#ifdef SNAPSHOT_POOL
	snapData.resizeDiscard(sizeof(PoolInfo)*SNAPSHOT_POOL);
	PoolInfo* &snapshots = *(PoolInfo**)module->getGlobal("g_snapshots").getMutablePtr();
	snapshots = (PoolInfo*)snapData.getMutableCudaPtr();
	snapData.clearRange32(0, 0, SNAPSHOT_POOL*sizeof(PoolInfo)); // Mark all tasks as empty (important for debug)
#endif
#ifdef SNAPSHOT_WARP
	snapData.resizeDiscard(sizeof(WarpInfo)*SNAPSHOT_WARP*NUM_WARPS);
	WarpInfo* &snapshots = *(WarpInfo**)module->getGlobal("g_snapshots").getMutablePtr();
	snapshots = (WarpInfo*)snapData.getMutableCudaPtr();
	snapData.clearRange32(0, 0, SNAPSHOT_WARP*NUM_WARPS*sizeof(WarpInfo)); // Mark all tasks as empty (important for debug)
#endif
}

//------------------------------------------------------------------------

void printSnapshots(Buffer &snapData)
{
#ifdef SNAPSHOT_POOL
	PoolInfo* snapshots = (PoolInfo*)snapData.getPtr();

	if(snapshots[SNAPSHOT_POOL-1].pool != 0) // Full
		printf("\aSnapshot memory full!\n");

	long long int clockMin = snapshots[0].clockStart;
	long long int clockMax = 0;
	for(int i = 0; i < SNAPSHOT_POOL; i++)
	{
		if(snapshots[i].pool == 0)
		{
			clockMax = snapshots[i-1].clockEnd;
			break;
		}
	}

	ofstream snapfile("plots\\pool\\activity.dat");
	snapfile << "Snap#\tpool\t#tasks\t#active\t#chunks\tdepth\tclocks" << "\n";
	for(int i = 0; i < SNAPSHOT_POOL; i++)
	{
		if(snapshots[i].pool == 0)
			break;

		snapfile << i << "\t" << snapshots[i].pool << "\t" << snapshots[i].tasks << "\t" << snapshots[i].active << "\t" << snapshots[i].chunks << "\t" << snapshots[i].depth
			<< "\t" << snapshots[i].clockEnd - snapshots[i].clockStart << "\n";
	}
	snapfile.close();

	snapfile.open("plots\\pool\\activity_clockCor.dat");
	snapfile << "Snap#\tpool\t#tasks\t#active\t#chunks\tdepth\tclocks" << "\n";
	for(int i = 0; i < SNAPSHOT_POOL; i++)
	{
		if(snapshots[i].pool == 0)
			break;

		snapfile << (float)((long double)(snapshots[i].clockEnd - clockMin) / (long double)(clockMax - clockMin)) << "\t" << snapshots[i].pool << "\t" << snapshots[i].tasks << "\t"
			<< snapshots[i].active << "\t" << snapshots[i].chunks << "\t" << snapshots[i].depth	<< "\t" << snapshots[i].clockEnd - snapshots[i].clockStart << "\n";
	}

	snapfile.close();
#endif
#ifdef SNAPSHOT_WARP
	WarpInfo* snapshots = (WarpInfo*)snapData.getPtr();

	for(int w = 0; w < NUM_WARPS; w++)
	{
		if(snapshots[SNAPSHOT_WARP-1].reads != 0) // Full
			printf("\aSnapshot memory full for warp %d!\n", w);

		ostringstream filename;
		filename.fill('0');
		filename << "plots\\warps\\warp" << setw(3) << w << ".dat";
		//cout << filename.str() << "\n";
		ofstream snapfile(filename.str());

		snapfile << "Snap#\t#reads\t#rays\t#tris\ttype(leaf=8)\t#chunks\tpopCount\tdepth\tcDequeue\tcCompute\tstackTop\ttaskIdx" << "\n";
		for(int i = 0; i < SNAPSHOT_WARP; i++)
		{
			if(snapshots[i].reads == 0)
				break;

			if(snapshots[i].clockDequeue < snapshots[i].clockSearch || snapshots[i].clockFinished < snapshots[i].clockDequeue)
				cout << "Error timer for warp " << w << "\n";

			snapfile << i << "\t" << snapshots[i].reads << "\t" << snapshots[i].rays << "\t" << snapshots[i].tris << "\t" << snapshots[i].type << "\t"
				<< snapshots[i].chunks << "\t" << snapshots[i].popCount << "\t" << snapshots[i].depth << "\t" << (snapshots[i].clockDequeue - snapshots[i].clockSearch) << "\t"
				<< (snapshots[i].clockFinished - snapshots[i].clockDequeue) << "\t" << snapshots[i].stackTop << "\t" << snapshots[i].idx << "\n";
		}

		snapfile.close();
		snapshots += SNAPSHOT_WARP; // Next warp
	}
#endif
}

//------------------------------------------------------------------------

}