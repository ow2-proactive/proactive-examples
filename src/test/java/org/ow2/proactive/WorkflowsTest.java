package org.ow2.proactive;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.Matchers.emptyString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.ow2.proactive.scheduler.common.exception.JobCreationException;
import org.ow2.proactive.scheduler.common.job.JobVariable;
import org.ow2.proactive.scheduler.common.job.TaskFlowJob;
import org.ow2.proactive.scheduler.common.job.factories.JobFactory;
import org.ow2.proactive.scheduler.common.task.Task;
import org.ow2.proactive.scheduler.core.properties.PASchedulerProperties;


@RunWith(Parameterized.class)
public class WorkflowsTest {

    private final static String ICON_URL_PATH_PREFIX = "/automation-dashboard/";

    private final static String WORKFLOW_ICON_KEY_NAME = "workflow.icon";

    private final static String TASK_ICON_KEY_NAME = "task.icon";

    private final static String CATALOG_OBJECT_DIR_PATH = "resources/catalog";

    private final static String METADATA_JSON_FILE = "METADATA.json";

    private final static String BOOLEAN_MODEL = "PA:Boolean";

    private final static String PCW_RULE_NAME = "rule";

    private final static String CATALOG_KEY_NAME = "catalog";

    private final static String OBJECTS_KEY_NAME = "objects";

    private final static String METADATA_KEY_NAME = "metadata";

    private final static String FILE_KEY_NAME = "file";

    private final static String WORKFLOW_KIND_KEY_NAME = "kind";

    private final static String WORKFLOW_KIND_VALUE = "workflow";

    public static final String WORKFLOW_NAME_PATTERN = "^(?:[A-Z\\d][a-zA-Z\\d\\.\\s]*)(?:[_A-Z\\d\\$][a-zA-Z\\d{}]*)*$";

    private final String filePath;

    private TaskFlowJob job = null;

    private static boolean isPackageDirIncludingCatalogObjects(Path packagePath) {
        return Files.isDirectory(packagePath) && Files.exists(Paths.get(packagePath.toString(), METADATA_JSON_FILE)) &&
               Files.exists(Paths.get(packagePath.toString(), CATALOG_OBJECT_DIR_PATH));
    }

    private static boolean isWorkflowKind(Path resourcePath){
        if(!resourcePath.toString().endsWith(".xml")){
            return false;
        }
        JSONObject jsonObject = getMetadataRootObject(resourcePath.getParent().getParent().getParent());
        JSONObject catalog = (JSONObject) jsonObject.get(CATALOG_KEY_NAME);
        if (catalog != null) {
            JSONArray objects = (JSONArray) catalog.get(OBJECTS_KEY_NAME);
            Optional<JSONObject> catalogObjectJson = objects.stream().filter(objectJsonObject -> (resourcePath.toString().contains(((JSONObject) objectJsonObject).get(FILE_KEY_NAME).toString()))).findAny();
            if (catalogObjectJson.isPresent()) {
                return ((String)((JSONObject)(catalogObjectJson.get()).get(METADATA_KEY_NAME)).get(WORKFLOW_KIND_KEY_NAME)).toLowerCase().startsWith(WORKFLOW_KIND_VALUE);
            }
        }
        return false;
    }

    private static JSONObject getMetadataRootObject(Path packagePath) {
        String metadataJsonFilePath = new File(packagePath.toString(), METADATA_JSON_FILE).getAbsolutePath();
        try {
            return  (JSONObject) new JSONParser().parse(new FileReader(metadataJsonFilePath));
        } catch (IOException | ParseException e) {
            throw new RuntimeException(e);
        }
    }

    public WorkflowsTest(String filePath) {
        this.filePath = filePath;
    }

    @Before
    public void init() throws Exception {
        PASchedulerProperties.CATALOG_REST_URL.updateProperty("http://localhost:8080/catalog");
        JobFactory factory = JobFactory.getFactory();
        this.job = (TaskFlowJob) factory.createJob(this.filePath);
    }

    @Parameterized.Parameters(name = "{index}: testing workflow - {0}")
    public static Collection<String> data() throws IOException {
        return Files.list(Paths.get("build"))
                    .filter(packagePath -> isPackageDirIncludingCatalogObjects(packagePath))
                    .map(packagePath -> Paths.get(packagePath.toString(), CATALOG_OBJECT_DIR_PATH))
                    .flatMap(resourcesPath -> {
                        try {
                            return Files.list(resourcesPath)
                                        .filter(file -> isWorkflowKind(file))
                                        .filter(file -> !file.toString().toLowerCase().contains(PCW_RULE_NAME));
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .map(workflowPath -> workflowPath.toString())
                    .collect(Collectors.toList());
    }

    @Test
    public void isWorkflowWellDefinedTest()
            throws IOException, TransformerException, ParserConfigurationException, JobCreationException {

        Map<String, JobVariable> jobVariables = this.job.getVariables();
        jobVariables.entrySet().stream().forEach(map -> {
            if (Arrays.asList("false", "true").contains(map.getValue().getValue().toLowerCase())) {
                assertThat("The wf variable: " + map.getValue().getName() + " MUST HAVE a boolean model: " +
                           BOOLEAN_MODEL,
                           map.getValue().getModel().toLowerCase(),
                           is(BOOLEAN_MODEL.toLowerCase()));
            }
        });

        String workflowName = this.job.getName();
        assertTrue("The workflow name " + workflowName +
                   " is invalid! Try an underscore-spaced name with Capitals or digits (e.g. Workflow_Name but not workflow_name)",
                   workflowName.matches(WORKFLOW_NAME_PATTERN));

        // Check mandatory generic information
        String workflowIconValue = this.job.getGenericInformation().get(WORKFLOW_ICON_KEY_NAME);
        assertThat("The wf MUST HAVE a Workflow Generic Information: " + WORKFLOW_ICON_KEY_NAME,
                   workflowIconValue,
                   notNullValue());
        assertThat("The wf MUST HAVE a meaningful Icon in a Workflow Generic Information: " + WORKFLOW_ICON_KEY_NAME,
                   workflowIconValue,
                   is(not(emptyString())));

        // 2. URL of this icon MUST reference a local file
        assertThat("URL of the icon " + WORKFLOW_ICON_KEY_NAME + " MUST reference a local file",
                   workflowIconValue,
                   startsWith(ICON_URL_PATH_PREFIX));

        // 3. If a workflow has a single task, this task MUST HAVE a Task Generic Information "task.icon" with the same icon as the Workflow
        List<Task> tasks = this.job.getTasks();
        if (tasks.size() == 1) {
            Task singleTask = tasks.get(0);
            String taskIconValue = singleTask.getGenericInformation().get(TASK_ICON_KEY_NAME);

            assertThat("The icon value in generic info for task should exist: " + TASK_ICON_KEY_NAME,
                       taskIconValue,
                       notNullValue());
            assertThat("The task MUST HAVE a Task Generic Information " + TASK_ICON_KEY_NAME +
                       " with the same icon as the Workflow: " + WORKFLOW_ICON_KEY_NAME,
                       taskIconValue,
                       equalTo(workflowIconValue));

        }
    }
}
