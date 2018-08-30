package org.ow2.proactive;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;


@RunWith(Parameterized.class)
public class MetadataJsonFilesTest {

    private final static String METADATA_JSON_FILE_NAME = "METADATA.json";

    private final static String DATASPACE_KEY_NAME = "dataspace";

    private final static String FILES_KEY_NAME = "files";

    private final static String CATALOG_KEY_NAME = "catalog";

    private final static String OBJECTS_KEY_NAME = "objects";

    private final static String FILE_KEY_NAME = "file";

    private final static String TEST_KEY_NAME = "test";

    private final static String SCENARIOS_KEY_NAME = "scenarios";

    private final String packageDirPath;

    private JSONParser parser;

    private static boolean isPackageDir(Path packagePath) {
        return Files.isDirectory(packagePath) &&
                Files.exists(Paths.get(packagePath.toString(), METADATA_JSON_FILE_NAME));
    }

    public MetadataJsonFilesTest(String packageDirPath) {
        this.packageDirPath = packageDirPath;
    }

    @Before
   public void init() {
        this.parser = new JSONParser();
    }


    @Parameterized.Parameters(name = "{index}: testing metadataJsonFile - {0}")
    public static List<String> data() throws IOException {
        return Files.list(Paths.get(""))
                                  .filter(packagePath -> isPackageDir(packagePath))
                                  .map(packagePath -> packagePath.toString())
                                  .collect(Collectors.toList());
    }

    @Test
    public void doResourcesExistTest()
            throws ParseException, IOException {

        String metadataJsonFilePath = new File(this.packageDirPath, METADATA_JSON_FILE_NAME).getAbsolutePath();
        JSONObject jsonObject = (JSONObject) this.parser.parse(new FileReader(metadataJsonFilePath));

        // Test if test scenarios are there
        JSONObject test =  (JSONObject) jsonObject.get(TEST_KEY_NAME);
        if (test != null) {
            String scenariosFilePath = (String) test.get(SCENARIOS_KEY_NAME);
            File scenariosFile = new File(this.packageDirPath, scenariosFilePath);
            Assert.assertTrue(scenariosFile + " does not exist!", scenariosFile.exists());
        }

        // Test if all dataspace resources are there
        JSONObject dataspace =  (JSONObject) jsonObject.get(DATASPACE_KEY_NAME);
        if (dataspace != null) {
            JSONArray filePaths = (JSONArray) dataspace.get(FILES_KEY_NAME);
            filePaths.forEach( dataspaceFilePath -> {
                    File dataspaceFile = new File(this.packageDirPath, (String)dataspaceFilePath);
                    Assert.assertTrue(dataspaceFile + " does not exist!", dataspaceFile.exists());});
        }

        // Test if all object resources are there
        JSONObject catalog =  (JSONObject) jsonObject.get(CATALOG_KEY_NAME);
        if (catalog != null) {
            JSONArray objects =  (JSONArray) catalog.get(OBJECTS_KEY_NAME);
            objects.forEach( objectJsonObject -> {
                String objectFilePath = (String)((JSONObject) objectJsonObject).get(FILE_KEY_NAME);
                File objectFile = new File(this.packageDirPath, objectFilePath);
                Assert.assertTrue(objectFile + " does not exist!", objectFile.exists());});
        }
    }
}
