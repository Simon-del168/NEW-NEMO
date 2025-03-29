package com.google.android.exoplayer2.testutil;

import static org.robolectric.Shadows.shadowOf;
import static org.robolectric.util.ReflectionHelpers.callInstanceMethod;

import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.os.MessageQueue;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import com.google.android.exoplayer2.util.Util;

import java.time.Duration;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import org.robolectric.annotation.Implementation;
import org.robolectric.annotation.Implements;
import org.robolectric.shadows.ShadowLooper;
import org.robolectric.shadows.ShadowMessageQueue;
import org.robolectric.util.Scheduler;

/** Collection of shadow classes used to run tests with Robolectric which require Loopers. */
public final class RobolectricUtil {

  private static final AtomicLong sequenceNumberGenerator = new AtomicLong(0);
  private static final int ANY_MESSAGE = Integer.MIN_VALUE;

  private RobolectricUtil() {}

  @Implements(Looper.class)
  public static final class CustomLooper extends ShadowLooper {

    private final PriorityBlockingQueue<PendingMessage> pendingMessages;
    private final CopyOnWriteArraySet<RemovedMessage> removedMessages;

    public CustomLooper() {
      pendingMessages = new PriorityBlockingQueue<>();
      removedMessages = new CopyOnWriteArraySet<>();
    }

    @Implementation
    public static void loop() {
      Looper looper = Looper.myLooper();
      if (shadowOf(looper) instanceof CustomLooper) {
        ((CustomLooper) shadowOf(looper)).doLoop();
      }
    }

    public Duration getLastScheduledTaskTime() {
      return Duration.ZERO; // 返回合适的 Duration 值
    }

    public Duration getNextScheduledTaskTime() {
      return Duration.ZERO; // 返回合适的 Duration 值
    }

    @Override
    public void quitUnchecked() {
      addPendingMessage(null, Long.MIN_VALUE); // 添加自定义退出逻辑
    }

    @Override
    public void pause() {
      // 根据测试需求实现暂停逻辑
    }

    @Override
    public boolean post(Runnable runnable, long delayMillis) {
      runnable.run(); // 简单地执行任务，或根据需要实现复杂的调度逻辑
      return true; // 返回 true 表示任务已成功调度
    }

    @Override
    public boolean postAtFrontOfQueue(Runnable runnable) {
      runnable.run(); // 简单地执行任务，或根据需要实现复杂的调度逻辑
      return true; // 返回 true 表示任务成功添加到队列前
    }

    @Override
    public void runOneTask() {
      // 从队列中取出并运行一个任务
      PendingMessage pendingMessage = pendingMessages.poll();
      if (pendingMessage != null && pendingMessage.message != null) {
        // 调用消息的目标处理
        Handler target = pendingMessage.message.getTarget();
        if (target != null) {
          target.dispatchMessage(pendingMessage.message);
        }
      }
    }

    @Override
    public void runToNextTask() {
      // 取出并执行队列中的下一个任务
      PendingMessage pendingMessage = pendingMessages.poll();
      if (pendingMessage != null && pendingMessage.message != null) {
        // 调用消息的目标处理
        Handler target = pendingMessage.message.getTarget();
        if (target != null) {
          target.dispatchMessage(pendingMessage.message);
        }
      }
    }

    @Override
    public void runToEndOfTasks() {
      // 运行所有队列中的任务直到为空
      while (!pendingMessages.isEmpty()) {
        runOneTask();
      }
    }

    @Override
    public void idleConstantly(boolean shouldIdle) {
      // 实现持续空闲或恢复的逻辑
      if (shouldIdle) {
        // 开始持续空闲，可能停止处理新任务
      } else {
        // 停止持续空闲，恢复正常任务处理
      }
    }

    private void addPendingMessage(@Nullable Message message, long when) {
      pendingMessages.put(new PendingMessage(message, when));
    }

    private void removeMessages(Handler handler, int what, Object object) {
      RemovedMessage newRemovedMessage = new RemovedMessage(handler, what, object);
      removedMessages.add(newRemovedMessage);
      for (RemovedMessage removedMessage : removedMessages) {
        if (removedMessage != newRemovedMessage
                && removedMessage.handler == handler
                && removedMessage.what == what
                && removedMessage.object == object) {
          removedMessages.remove(removedMessage);
        }
      }
    }

    private void doLoop() {
      boolean wasInterrupted = false;
      while (true) {
        try {
          PendingMessage pendingMessage = pendingMessages.take();
          if (pendingMessage.message == null) {
            // Null message is signal to end message loop.
            return;
          }
          callInstanceMethod(pendingMessage.message, "markInUse");
          Handler target = pendingMessage.message.getTarget();
          if (target != null) {
            boolean isRemoved = false;
            for (RemovedMessage removedMessage : removedMessages) {
              if (removedMessage.handler == target
                      && (removedMessage.what == ANY_MESSAGE
                      || removedMessage.what == pendingMessage.message.what)
                      && (removedMessage.object == null
                      || removedMessage.object == pendingMessage.message.obj)
                      && pendingMessage.sequenceNumber < removedMessage.sequenceNumber) {
                isRemoved = true;
              }
            }
            if (!isRemoved) {
              try {
                if (wasInterrupted) {
                  wasInterrupted = false;
                  Thread.currentThread().interrupt();
                }
                target.dispatchMessage(pendingMessage.message);
              } catch (Throwable t) {
                Looper.getMainLooper().getThread().interrupt();
                throw t;
              }
            }
          }
          if (Util.SDK_INT >= 21) {
            callInstanceMethod(pendingMessage.message, "recycleUnchecked");
          } else {
            callInstanceMethod(pendingMessage.message, "recycle");
          }
        } catch (InterruptedException e) {
          wasInterrupted = true;
        }
      }
    }
  }

  @Implements(MessageQueue.class)
  public static final class CustomMessageQueue extends ShadowMessageQueue {

    private final Thread looperThread;

    public CustomMessageQueue() {
      looperThread = Thread.currentThread();
    }

    @Override
    public Message getHead() {
      return null; // 返回合适的头部消息或 null
    }

    @Override
    public void setHead(Message msg) {
      // 实现 setHead 的逻辑
    }

    @Override
    public void setScheduler(Scheduler scheduler) {
      // 实现 setScheduler 的逻辑
    }

    @Override
    public Scheduler getScheduler() {
      return null; // 返回当前的调度器，或根据需要调整
    }

    @Implementation
    public boolean enqueueMessage(Message msg, long when) {
      Looper looper = ShadowLooper.getLooperForThread(looperThread);
      if (shadowOf(looper) instanceof CustomLooper
              && shadowOf(looper) != ShadowLooper.getShadowMainLooper()) {
        ((CustomLooper) shadowOf(looper)).addPendingMessage(msg, when);
      } else {
        // 手动处理消息添加逻辑
        return false; // 如果不能直接调用父类的方法
      }
      return true;
    }

    @Override
    public void reset() {
      // 实现 reset 方法的逻辑，如果需要
    }

    @Implementation
    public void removeMessages(Handler handler, int what, Object object) {
      Looper looper = ShadowLooper.getLooperForThread(looperThread);
      if (shadowOf(looper) instanceof CustomLooper
              && shadowOf(looper) != ShadowLooper.getShadowMainLooper()) {
        ((CustomLooper) shadowOf(looper)).removeMessages(handler, what, object);
      }
    }

    @Implementation
    public void removeCallbacksAndMessages(Handler handler, Object object) {
      Looper looper = ShadowLooper.getLooperForThread(looperThread);
      if (shadowOf(looper) instanceof CustomLooper
              && shadowOf(looper) != ShadowLooper.getShadowMainLooper()) {
        ((CustomLooper) shadowOf(looper)).removeMessages(handler, ANY_MESSAGE, object);
      }
    }
  }

  private static final class PendingMessage implements Comparable<PendingMessage> {
    public final @Nullable Message message;
    public final long when;
    public final long sequenceNumber;

    public PendingMessage(@Nullable Message message, long when) {
      this.message = message;
      this.when = when;
      sequenceNumber = sequenceNumberGenerator.getAndIncrement();
    }

    @Override
    public int compareTo(@NonNull PendingMessage other) {
      int res = Util.compareLong(this.when, other.when);
      if (res == 0 && this != other) {
        res = Util.compareLong(this.sequenceNumber, other.sequenceNumber);
      }
      return res;
    }
  }

  private static final class RemovedMessage {
    public final Handler handler;
    public final int what;
    public final Object object;
    public final long sequenceNumber;

    public RemovedMessage(Handler handler, int what, Object object) {
      this.handler = handler;
      this.what = what;
      this.object = object;
      this.sequenceNumber = sequenceNumberGenerator.get();
    }
  }
}
