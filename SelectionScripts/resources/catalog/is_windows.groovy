/**
 * Script which verifies that the current node runs in a windows machine
 */

import org.ow2.proactive.utils.OperatingSystem;
import org.ow2.proactive.utils.OperatingSystemFamily;

String osName = System.getProperty("os.name");
println "Node operating system : " + osName;
OperatingSystem operatingSystem = OperatingSystem.resolveOrError(osName);
OperatingSystemFamily family = operatingSystem.getFamily();

switch (family) {
    case OperatingSystemFamily.WINDOWS:
        selected = true;
        break;
    default:
        selected = false;
}