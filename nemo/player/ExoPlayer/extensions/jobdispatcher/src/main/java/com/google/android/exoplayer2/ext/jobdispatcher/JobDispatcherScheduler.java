package com.google.android.exoplayer2.ext.jobdispatcher;

import android.content.Context;
import android.content.Intent;
import androidx.annotation.NonNull;
import androidx.work.Constraints;
import androidx.work.Data;
import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkManager;
import androidx.work.WorkRequest;
import androidx.work.WorkerParameters;
import com.google.android.exoplayer2.scheduler.Requirements;
import com.google.android.exoplayer2.scheduler.Scheduler;
import com.google.android.exoplayer2.util.Assertions;
import com.google.android.exoplayer2.util.Log;
import com.google.android.exoplayer2.util.Util;

/**
 * A {@link Scheduler} that uses {@link WorkManager}. This Scheduler no longer requires
 * {@link JobDispatcherSchedulerService} to be added to the manifest.
 */
public final class JobDispatcherScheduler implements Scheduler {

  private static final String TAG = "JobDispatcherScheduler";
  private static final String KEY_SERVICE_ACTION = "service_action";
  private static final String KEY_SERVICE_PACKAGE = "service_package";
  private static final String KEY_REQUIREMENTS = "requirements";

  private final String jobTag;
  private final Context context;

  /**
   * @param context A context.
   * @param jobTag  A tag for jobs scheduled by this instance. If the same tag was used by a previous
   *                instance, anything scheduled by the previous instance will be canceled by this instance if
   *                {@link #schedule(Requirements, String, String)} or {@link #cancel()} are called.
   */
  public JobDispatcherScheduler(Context context, String jobTag) {
    this.context = context.getApplicationContext();
    this.jobTag = jobTag;
  }

  @Override
  public boolean schedule(Requirements requirements, String serviceAction, String servicePackage) {
    Constraints constraints = buildConstraints(requirements);

    // Prepare data to pass to the Worker
    Data inputData = new Data.Builder()
            .putString(KEY_SERVICE_ACTION, serviceAction)
            .putString(KEY_SERVICE_PACKAGE, servicePackage)
            .putInt(KEY_REQUIREMENTS, requirements.getRequirementsData())
            .build();

    // Create a one-time work request with constraints
    WorkRequest workRequest = new OneTimeWorkRequest.Builder(JobDispatcherSchedulerWorker.class)
            .setConstraints(constraints)
            .setInputData(inputData)
            .addTag(jobTag)
            .build();

    // Enqueue the work
    WorkManager.getInstance(context).enqueue(workRequest);
    logd("Scheduling job with WorkManager: " + jobTag);
    return true;
  }

  @Override
  public boolean cancel() {
    WorkManager.getInstance(context).cancelAllWorkByTag(jobTag);
    logd("Canceling job with WorkManager: " + jobTag);
    return true;
  }

  private Constraints buildConstraints(Requirements requirements) {
    Constraints.Builder builder = new Constraints.Builder();

    switch (requirements.getRequiredNetworkType()) {
      case Requirements.NETWORK_TYPE_ANY:
        builder.setRequiredNetworkType(androidx.work.NetworkType.CONNECTED);
        break;
      case Requirements.NETWORK_TYPE_UNMETERED:
        builder.setRequiredNetworkType(androidx.work.NetworkType.UNMETERED);
        break;
      default:
        // If no network requirement, no need to set constraint
        break;
    }

    if (requirements.isIdleRequired()) {
      builder.setRequiresDeviceIdle(true);
    }
    if (requirements.isChargingRequired()) {
      builder.setRequiresCharging(true);
    }

    return builder.build();
  }

  private static void logd(String message) {
    Log.d(TAG, message);
  }

  /**
   * Worker class that starts the target service if the requirements are met.
   */
  public static class JobDispatcherSchedulerWorker extends androidx.work.Worker {

    public JobDispatcherSchedulerWorker(@NonNull Context context, @NonNull WorkerParameters params) {
      super(context, params);
    }

    @NonNull
    @Override
    public Result doWork() {
      logd("JobDispatcherSchedulerWorker is started");

      Data inputData = getInputData();
      String serviceAction = inputData.getString(KEY_SERVICE_ACTION);
      String servicePackage = inputData.getString(KEY_SERVICE_PACKAGE);
      int requirementsData = inputData.getInt(KEY_REQUIREMENTS, 0);

      Requirements requirements = new Requirements(requirementsData);
      if (requirements.checkRequirements(getApplicationContext())) {
        logd("Requirements are met");
        Assertions.checkNotNull(serviceAction, "Service action missing.");
        Assertions.checkNotNull(servicePackage, "Service package missing.");

        Intent intent = new Intent(serviceAction).setPackage(servicePackage);
        logd("Starting service action: " + serviceAction + " package: " + servicePackage);
        Util.startForegroundService(getApplicationContext(), intent);
        return Result.success();
      } else {
        logd("Requirements are not met");
        return Result.retry();
      }
    }

    private static void logd(String message) {
      Log.d(TAG, message);
    }
  }
}
