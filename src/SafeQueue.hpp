// Implementation from https://stackoverflow.com/questions/15278343/c11-thread-safe-queue
#ifndef THREAD_SAFE_QUEUE_HPP
#define THREAD_SAFE_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace mysdk {

// A threadsafe-queue.
template <class T>
class SafeQueue
{
public:
  SafeQueue(void)
    : q()
    , m()
    , c()
  {}

  ~SafeQueue(void)
  {}

  // Check is queue empty
  bool empty() const
  {
    std::lock_guard<std::mutex> lock(m);
    return q.empty();
  }

  // Return queue size
  size_t size() const
  {
    std::lock_guard<std::mutex> lock(m);
    return q.size();
  }

  // Add an element to the queue.
  void enqueue(T t)
  {
    std::lock_guard<std::mutex> lock(m);
    q.push(t);
    c.notify_one();
  }

  // Get the "front"-element.
  // If the queue is empty, wait till a element is avaiable.
  std::optional<T> dequeue(void)
  {
    std::unique_lock<std::mutex> lock(m);
    std::optional<T> result;

    if(q.empty())
     return result;

    result.emplace(q.front());
    q.pop();
    return result;
  }

  // Get the "front"-element.
  // If the queue is empty, wait till a element is avaiable.
  T dequeue_wait(void)
  {
    std::unique_lock<std::mutex> lock(m);
    while(q.empty())
    {
      // release lock as long as the wait and reaquire it afterwards.
      c.wait(lock);
    }
    T val = q.front();
    q.pop();
    return val;
  }

private:
  std::queue<T> q;
  mutable std::mutex m;
  std::condition_variable c;
};

}

#endif
