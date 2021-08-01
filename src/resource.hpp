#ifndef _resource_hpp
#define _resource_hpp

#include <iostream>
#include <optional>

/* automagic deleter */
template<class T, typename F>
struct resource
{
  std::optional<T> x;
  F f;

  resource();

  resource(const resource &) = delete;
  resource &operator=(const resource &) = delete;

  resource(resource &&other)
    : x(std::move(other.x))
    , f(std::move(other.f))
  {
    other.x.reset(); // this is needed if the other thing is a wrapped primitive
                     // value
  }

  resource &operator=(resource &&other)
  {
    x.swap(other.x);
    std::swap(f, other.f);
  }

  resource(const T &x, F f)
    : x(x)
    , f(f)
  {}

  resource(T &&x, F f)
    : x(std::move(x))
    , f(f)
  {}

  T &operator=(const T &other)
  {
    x = other;
    return *this;
  }
  T &operator=(T &&other)
  {
    x = std::move(other);
    return *this;
  }

  operator T &() { return *x; }
  operator const T &() const { return *x; }
  T &operator*() { return *x; }
  const T &operator*() const { return *x; }
  T *operator->() { return &(*x); }
  const T *operator->() const { return &(*x); }

  ~resource()
  {
    std::cerr << "resource erase: " << typeid(T).name() << " (" << x.has_value()
              << ")" << std::endl;
    if (x.has_value())
      f(*x);
  }
};

#endif // _resource_hpp
