# make sure that there are no errors raised in the doc strings during the mkdocs
# build

SITEDIR=_build/html

TS=$(grep -R "Traceback" $SITEDIR | wc -l)
if (($TS > 0)); then
    echo "Traceback found in one of the doc strings!"
    echo $(grep -R "Traceback" $SITEDIR)
    exit 1
fi
