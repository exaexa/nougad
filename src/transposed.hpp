
#ifndef _transposed_hpp
#define _transposed_hpp

/*
 * transposed matrix view -- iterating on transposed(ptr,rows,cols) that is
 * stored by columns outputs (or ingests) the numbers as if it was read by
 * rows.
 */

template<typename T>
struct transposed
{

  T *ptr;
  const size_t a, b;

  transposed(T *ptr, size_t a, size_t b)
    : ptr(ptr)
    , a(a)
    , b(b)
  {}

  size_t size() const { return a * b; }

  struct tr_iter
  {
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using reference = T &;

    size_t off;
    transposed base;
    tr_iter(size_t off, transposed base)
      : off(off)
      , base(base)
    {}

    inline reference operator*() const
    {
      const size_t blk = off / base.b;
      const size_t item = off % base.b;
      return base.ptr[blk + base.a * item];
    }

    tr_iter &operator++()
    {
      ++off;
      return *this;
    }

    ptrdiff_t operator-(tr_iter other) const { return off - other.off; }
    bool operator==(tr_iter other) const { return off == other.off; }
    bool operator!=(tr_iter other) const { return off != other.off; }
    bool operator<(tr_iter other) const { return off < other.off; }
  };

  tr_iter begin() const { return tr_iter(0, *this); }
  tr_iter end() const { return tr_iter(a * b, *this); }
  tr_iter at(size_t ia, size_t ib) const { return tr_iter(ib + b * ia, *this); }
};

#endif // _transposed_hpp
