package org.ow2.proactive;

import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.Assert.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.junit.Test;


public class OrderedBucketListTest {
    private final static String ORDERED_BUCKET_LIST_FILE_NAME = "ordered_bucket_list";

    @Test
    public void testOrderedBucketList() throws IOException {

        List<String> fileLinesList = Files.readAllLines(Paths.get(ORDERED_BUCKET_LIST_FILE_NAME));

        assertThat("The ordered bucket list file should not be empty: " + ORDERED_BUCKET_LIST_FILE_NAME,
                   fileLinesList,
                   is(not(empty())));

        assertThat("The ordered bucket list file should contain only 1 line: ", fileLinesList.size(), is(1));

        assertThat("The ordered bucket list should not contain spaces: ",
                   fileLinesList.get(0).contains(" "),
                   is(false));
    }
}
