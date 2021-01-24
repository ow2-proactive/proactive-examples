signal = args[0]


if (signalapi.isReceived(signal)){
    println("Signal " + signal + " is received")
} else {
    println("Signal " + signal + " is not received")
}
