import com.google.common.base.Splitter;

if (variables.get("PA_TASK_ITERATION")==0){

    // Read the variable SIGNALS
    signals = variables.get("SIGNALS")

    // Split the value of the variable SIGNALS and transform it into a list
    Set signalsSet = new HashSet<>(Splitter.on(',').trimResults().omitEmptyStrings().splitToList(signals))

    // Send a ready notification for each signal in the set
    println("Ready for signals "+ signalsSet)
    signalsSet.each {
        signal -> signalapi.readyForSignal(signal)
    }

    // Add the signals set as a variable to be used by next tasks
    variables.put("SIGNALS_SET", signalsSet)
}

//Read the variable SIGNALS_SET
Set signalsSet =  variables.get("SIGNALS_SET")

// Check whether one signal among those specified as input is received
println("Checking whether one signal in the set "+ signalsSet +" is received")
receivedSignals = signalapi.checkForSignals(signalsSet)

// If a signal is received, remove ready signals and break the loop, else sleep 10 seconds then restart
if (receivedSignals != null && !receivedSignals.isEmpty()){

    // remove ready signals
    signalapi.removeManySignals(new HashSet<>(signalsSet.collect { signal -> "ready_"+signal }))

    // print the received signals
    println("Received signals: "+ receivedSignals.toString())
    result = receivedSignals.keySet().toString()

} else {
    result = null
}