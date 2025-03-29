package com.google.android.exoplayer2.ext.flac;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.google.android.exoplayer2.testutil.ExtractorAsserts;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Unit test for {@link FlacExtractor}.
 */
@RunWith(AndroidJUnit4.class)
public class FlacExtractorTest {

  @Before
  public void setUp() throws Exception {
    if (!FlacLibrary.isAvailable()) {
      throw new AssertionError("Flac library not available.");
    }
  }

  @Test
  public void testExtractFlacSample() throws Exception {
    ExtractorAsserts.assertBehavior(
            FlacExtractor::new, "bear.flac", InstrumentationRegistry.getInstrumentation().getContext());
  }

  @Test
  public void testExtractFlacSampleWithId3Header() throws Exception {
    ExtractorAsserts.assertBehavior(
            FlacExtractor::new, "bear_with_id3.flac", InstrumentationRegistry.getInstrumentation().getContext());
  }
}
