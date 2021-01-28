import com.google.common.base.Splitter;

signals = args[0]

signalsList = Splitter.on(',').trimResults().omitEmptyStrings().splitToList(signals)

println("Waiting for signal "+ signalsList)

signalapi.waitForAny(signalsList)
