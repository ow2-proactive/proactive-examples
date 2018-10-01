/**
 * Script which verifies that the current node's host matches the given IP address.
 * The IP can be given as x.x.x.x or using the token * to match a network for example. (ie x.x.x.*)
 *
 * Arguments:
 * machine ip address
 */

import com.google.common.base.Strings;
import org.ow2.proactive.scripting.helper.selection.SelectionUtils
import java.net.NetworkInterface;
import java.util.Collections

if (args.length != 1) {
    println "Incorrect number of arguments, expected 1, received " + args.length;
    selected = false;
    return;
}

ipAddress = args[0]

if (Strings.isNullOrEmpty(ipAddress)) {
    println "Given ip address was empty";
    selected = false;
    return;
}

ipAddress = ipAddress.trim()

println "Ip addresses " + Collections.list(NetworkInterface.getNetworkInterfaces()).collect { Collections.list(it.getInetAddresses()).collect { it.getHostAddress() } } + " (expected : " + ipAddress + ")";

selected = SelectionUtils.checkIp(ipAddress)


