#!/bin/bash
while true
do
	curl -X POST -H "Content-Type: application/json" -k -d '{"data": {"ndarray": [[123456, 13.0, 33.0, 6.0, 287.0, 22.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 123277432974]]}}' https://vmx-eea166.ete.ka.sw.ericsson.se:32222/model-endpoints/com-ericsson-eea-ubclustering 
	sleep $((RANDOM % 10))
done