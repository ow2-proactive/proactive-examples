/**
 * Pre script which allows to propagate an error which occurred in a parent task.
 * If a parent task contained an error, the current task will throw an IllegalStateException with message:
 * Parent task id=task_id (task_name) failed
 */

results.each { res ->
    if (res.hadException()) {
        throw new IllegalStateException("Parent task id=" + res.getTaskId().value() + " (" + res.getTaskId().getReadableName() + ") failed")
    }
}
