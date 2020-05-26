INSTANCE_NAME=$variables_INSTANCE_NAME

if [ "$(docker ps -a --format '{{.Names}}' | grep "^$INSTANCE_NAME$")" ]; then
 RUNNING=$(docker inspect --format="{{ .State.Running }}" $INSTANCE_NAME 2> /dev/null)
 STOPPED=$(docker inspect --format="{{ .State.Status }}" $INSTANCE_NAME 2> /dev/null)
	if [ "$RUNNING" == "true" ]; then
   		echo docker container: "$INSTANCE_NAME" is running
        echo $INSTANCE_NAME > $INSTANCE_NAME"_status"
	elif [ "$STOPPED" == "exited" ]; then
		echo Starting docker container: "$INSTANCE_NAME"
        INSTANCE_STATUS=$(docker start $INSTANCE_NAME 2>&1)
        echo $INSTANCE_STATUS > $INSTANCE_NAME"_status"
	fi
else
    echo Error: No such container: "$INSTANCE_NAME" > $INSTANCE_NAME"_status"
fi