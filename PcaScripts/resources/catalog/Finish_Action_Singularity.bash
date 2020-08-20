INSTANCE_NAME=$variables_INSTANCE_NAME
echo Removing singularity container: "$variables_INSTANCE_NAME"

if [ "$(singularity instance list | grep "$INSTANCE_NAME")" ]; then
    singularity instance stop $INSTANCE_NAME
    echo $INSTANCE_NAME > $variables_INSTANCE_NAME"_status"
else
	echo "[WARNING] sigularity container: $INSTANCE_NAME is already removed."
fi