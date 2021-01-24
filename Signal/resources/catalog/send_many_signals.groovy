import com.google.common.base.Splitter;

signals = args[0]

signalsList = Splitter.on(',').trimResults().omitEmptyStrings().splitToList(signals)

signalapi.sendManySignals(signalsList)
