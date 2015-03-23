#!/bin/sh
# Very simple script for tabelling benchmark results.

PROJECTDIR=..
cd $PROJECTDIR
PROJECTDIR="."
TESTDIR="tests"
LOGDIR="logs"

TEST=mv1
SCENES="Armadillo/armadillo.obj"

OUTDIR=$TESTDIR/results/`date +%Y-%m-%d`-$TEST

mkdir -p $OUTDIR

OUT=$OUTDIR/tests-$TEST.txt

Round2()
{
echo $1 | awk '{printf("%3.2f", $1 )}'
}

GetValue() {
V=`grep -A1 "#$1$" $2 | head -2 | tail -1`
if [ "$V" != "" ]; then
  echo $V
else
  echo "0"
fi  
}

Divide()
{
echo "$1 $2" | awk '{printf("%3.2f", $1/$2)}'
}

rm -f $OUT

for S in $SCENES; do
echo "------------------" >> $OUT
echo $S >> $OUT

LINE=""
f=$TESTDIR/$LOGDIR/$S-$TEST.log
#TRIANGLES=`GetValue SCENE_TRIANGLES $f`
SUM_RENDER_TIME=`GetValue SUM_RENDER_TIME $f`

LINE="$LINE $SUM_RENDER_TIME"
echo "$LINE" >> $OUT
echo "------------------" >> $OUT
done

cat $OUT
