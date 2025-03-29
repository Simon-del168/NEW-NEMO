package com.google.android.exoplayer2.ext.flac;

import static com.google.common.truth.Truth.assertThat;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.google.android.exoplayer2.extractor.SeekMap;
import com.google.android.exoplayer2.testutil.FakeExtractorInput;
import com.google.android.exoplayer2.testutil.TestUtil;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit test for {@link FlacBinarySearchSeeker}. */
@RunWith(AndroidJUnit4.class)
public final class FlacBinarySearchSeekerTest {

  private static final String NOSEEKTABLE_FLAC = "bear_no_seek.flac";
  private static final int DURATION_US = 2_741_000;

  @Before
  public void setUp() throws Exception {
    if (!FlacLibrary.isAvailable()) {
      throw new AssertionError("Flac library not available.");
    }
  }

  @Test
  public void testGetSeekMap_returnsSeekMapWithCorrectDuration()
          throws IOException, FlacDecoderException, InterruptedException {
    byte[] data = TestUtil.getByteArray(
            InstrumentationRegistry.getInstrumentation().getContext(), NOSEEKTABLE_FLAC);

    FakeExtractorInput input = new FakeExtractorInput.Builder().setData(data).build();
    FlacDecoderJni decoderJni = new FlacDecoderJni();
    decoderJni.setData(input);

    FlacBinarySearchSeeker seeker =
            new FlacBinarySearchSeeker(
                    decoderJni.decodeMetadata(), /* firstFramePosition= */ 0, data.length, decoderJni);

    SeekMap seekMap = seeker.getSeekMap();
    assertThat(seekMap).isNotNull();
    assertThat(seekMap.getDurationUs()).isEqualTo(DURATION_US);
    assertThat(seekMap.isSeekable()).isTrue();
  }

  @Test
  public void testSetSeekTargetUs_returnsSeekPending()
          throws IOException, FlacDecoderException, InterruptedException {
    byte[] data = TestUtil.getByteArray(
            InstrumentationRegistry.getInstrumentation().getContext(), NOSEEKTABLE_FLAC);

    FakeExtractorInput input = new FakeExtractorInput.Builder().setData(data).build();
    FlacDecoderJni decoderJni = new FlacDecoderJni();
    decoderJni.setData(input);
    FlacBinarySearchSeeker seeker =
            new FlacBinarySearchSeeker(
                    decoderJni.decodeMetadata(), /* firstFramePosition= */ 0, data.length, decoderJni);

    seeker.setSeekTargetUs(/* timeUs= */ 1000);
    assertThat(seeker.isSeeking()).isTrue();
  }
}
