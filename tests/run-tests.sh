#!/bin/bash
# Very simple script for running the benchmark.

PROJECTDIR=..
cd $PROJECTDIR
PROJECTDIR="."
TESTDIR="tests"
COMMAND="$PROJECTDIR/NTrace_Win32_Release.exe $PROJECTDIR/config.conf"
TEST=mv1

LOGDIR="logs"

DATASTRUCTURE="BVH"
BUILDER="SAHBVH"
SCENES="Armadillo/armadillo.obj"
#RAYTYPE="primary;AO;diffuse"
RAYTYPE="primary"
CAMERAS="GBSvz1V04qy/Ju69/21iChCz/idyKy10A0Kfx1pzUoy/DuY2/0aNqY10sZpuu/5/5f/0/"

ALLLOGS=""

run()
{
#rm -f $PROJECTDIR/ntrace.log
FILE=$TESTDIR/$LOGDIR/$S-$TEST
EXECUTE="$* -DApp.stats=$PROJECTDIR/$FILE.log -DBenchmark.screenshotName=${FILE}_kernel=%d_rt=%s_cam=%d.png"
$EXECUTE
ALLLOGS="$ALLLOGS $FILE"
}

# Launch over all scenes
for S in $SCENES; do
mkdir -p $TESTDIR/$LOGDIR/${S%/*}
echo Running test for $S
run $COMMAND -DApp.benchmark="true" -DBenchmark.scene="$PROJECTDIR/data/$S" -DBenchmark.camera=$CAMERAS -DBenchmark.screenshot="true" -DRenderer.dataStructure=$DATASTRUCTURE -DRenderer.builder=$BUILDER -DRenderer.rayType=$RAYTYPE
done

#OUTDIR=$TESTDIR/results/`date +%Y-%m-%d`-$TEST
#mkdir -p $OUTDIR
#zip $OUTDIR/`date +%Y-%m-%d`-$TEST.zip $ALLLOGS
