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
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Stream;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.ow2.proactive.scheduler.common.exception.JobCreationException;
import org.ow2.proactive.scheduler.common.job.TaskFlowJob;
import org.ow2.proactive.scheduler.common.job.factories.JobFactory;
import org.ow2.proactive.scheduler.common.task.Task;


/**
 * This test checks:
 * 1. Every single workflow of packages distributed by Activeeon (as all the workflows from proactive-examples), MUST HAVE a Workflow Generic Information "pa.action.icon" with a meaningful Icon
 * 2. URL of this icon MUST reference a local file
 * 3. If a workflow has a single task, this task MUST HAVE a Task Generic Information "task.icon" with the same icon as the Workflow
 */
@RunWith(Parameterized.class)
public class ProActiveExamplesValidityTest {

    private final static String ICON_URL_PATH_PREFIX = "/automation-dashboard/";

    private final static String JOB_ICON_KEY_NAME = "pca.action.icon";

    private final static String TASK_ICON_KEY_NAME = "task.icon";

    private final static String PATH_TO_WORKFLOWS = "resources/catalog";

    private final String filePath;

    public ProActiveExamplesValidityTest(String filePath) {
        this.filePath = filePath;
    }

    @Parameterized.Parameters(name = "{index}: testing workflow - {0}")
    public static Collection<String> data() throws IOException {

        List<String> files = new ArrayList<>();
        try (Stream<Path> rootFilesStream = Files.list(Paths.get(""))) {
            rootFilesStream.filter(entry -> Files.isDirectory(entry) &&
                                            Files.exists(Paths.get(entry.toString(), "METADATA.json")))
                           .map(directoryPath -> Paths.get(directoryPath.toString(), PATH_TO_WORKFLOWS))
                           .flatMap(resourcesPath -> {
                               try {
                                   return Files.list(resourcesPath).filter(file -> file.toString().endsWith(".xml"));
                               } catch (IOException e) {
                                   throw new RuntimeException(e);
                               }
                           })
                           .forEach(workflowPath -> files.add(workflowPath.toString()));
        }
        return files;
    }

    @Test
    public void testWfDescription()
            throws IOException, TransformerException, ParserConfigurationException, JobCreationException {
        JobFactory factory = JobFactory.getFactory();
        TaskFlowJob job = (TaskFlowJob) factory.createJob(filePath);

        /*
         * 1. Every single workflow of packages distributed by Activeeon (as all the workflows from
         * proactive-examples),
         * MUST HAVE a Workflow Generic Information "pa.action.icon" with a meaningful Icon
         */
        String pcaActionIconValue = job.getGenericInformation().get(JOB_ICON_KEY_NAME);
        assertThat("The wf MUST HAVE a Workflow Generic Information: " + JOB_ICON_KEY_NAME,
                   pcaActionIconValue,
                   notNullValue());
        assertThat("The wf MUST HAVE a meaningful Icon in a Workflow Generic Information: " + JOB_ICON_KEY_NAME,
                   pcaActionIconValue,
                   is(not(emptyString())));

        // 2. URL of this icon MUST reference a local file
        assertThat("URL of the icon " + JOB_ICON_KEY_NAME + " MUST reference a local file",
                   pcaActionIconValue,
                   startsWith(ICON_URL_PATH_PREFIX));

        // 3. If a workflow has a single task, this task MUST HAVE a Task Generic Information "task.icon" with the same icon as the Workflow
        List<Task> tasks = job.getTasks();
        if (tasks.size() == 1) {
            Task singleTask = tasks.get(0);
            String taskIconValue = singleTask.getGenericInformation().get(TASK_ICON_KEY_NAME);

            assertThat("The icon value in generic info for task should exist: " + TASK_ICON_KEY_NAME,
                       taskIconValue,
                       notNullValue());
            assertThat("The task MUST HAVE a Task Generic Information " + TASK_ICON_KEY_NAME +
                       " with the same icon as the Workflow: " + JOB_ICON_KEY_NAME,
                       taskIconValue,
                       equalTo(pcaActionIconValue));

        }
    }
}
