echo Removing docker container: "$variables_INSTANCE_NAME"
INSTANCE_NAME=$(docker rm -fv $variables_INSTANCE_NAME 2>&1)
echo $INSTANCE_NAME > $variables_INSTANCE_NAME"_status"