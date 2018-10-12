package org.ow2.proactive;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.Matchers.emptyString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.Assert.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;

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

    private final String filePath;

    private TaskFlowJob job = null;

    private static boolean isPackageDirIncludingCatalogObjects(Path packagePath) {
        return Files.isDirectory(packagePath) && Files.exists(Paths.get(packagePath.toString(), METADATA_JSON_FILE)) &&
               Files.exists(Paths.get(packagePath.toString(), CATALOG_OBJECT_DIR_PATH));
    }

    public WorkflowsTest(String filePath) {
        this.filePath = filePath;
    }

    @Before
    public void init() throws Exception {
        JobFactory factory = JobFactory.getFactory();
        this.job = (TaskFlowJob) factory.createJob(this.filePath);
    }

    @Parameterized.Parameters(name = "{index}: testing workflow - {0}")
    public static Collection<String> data() throws IOException {
        return Files.list(Paths.get(""))
                    .filter(packagePath -> isPackageDirIncludingCatalogObjects(packagePath))
                    .map(packagePath -> Paths.get(packagePath.toString(), CATALOG_OBJECT_DIR_PATH))
                    .flatMap(resourcesPath -> {
                        try {
                            return Files.list(resourcesPath)
                                        .filter(file -> file.toString().endsWith(".xml"))
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
                           BOOLEAN_MODEL, map.getValue().getModel(), is(BOOLEAN_MODEL));
            }
        });

        // Check mandatory generic informations
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
