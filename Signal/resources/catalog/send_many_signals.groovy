import com.google.common.base.Splitter;

signals = args[0]

Set signalsSet = new HashSet<>(Splitter.on(',').trimResults().omitEmptyStrings().splitToList(signals))

signalapi.sendManySignals(signalsSet)
