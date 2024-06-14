#!/bin/bash

while getopts "dsp" flag;do
    case "${flag}" in
      d) # handle data exporter
        echo "do data export"
        doDataExporter="true"
        ;;
      s) #handle Scheduler Service
        echo "do Scheduler Service"
        doSchedulerService="true"
        ;;
      p) #handle python stuff
        echo "do Python Stuff"
        doPython="true"
        ;;
    esac
done
echo "Value $doSchedulerService" . $doSchedulerService
if [ "$doDataExporter" = "true" ]; then
    echo "Start Data Export build"
    docker build --progress plain --pull --rm -f "csharp/DataExporter/Dockerfile" -t neichmann/dataexporter:latest "csharp"
    docker push neichmann/dataexporter:latest
fi

if [ "$doSchedulerService" = "true" ] ; then
    echo "Start Scheduler Service build"
    docker build --progress plain --pull --rm -f "csharp/SchedulerService/Dockerfile" -t neichmann/schedulerservice:latest "csharp"
    docker push neichmann/schedulerservice:latest
fi

if [ "$doPython" = "true" ] ; then
    docker build --progress plain --pull --rm -f "python/Dockerfile" -t neichmann/modeltrainer:latest "python"
    docker push neichmann/modeltrainer:latest
fi