import com.google.common.base.Splitter;

signals = args[0]

Set signalsSet = new HashSet<>(Splitter.on(',').trimResults().omitEmptyStrings().splitToList(signals))

println("Waiting for all the following signals: "+ signalsSet)

signalapi.waitForAll(signalsSet)




