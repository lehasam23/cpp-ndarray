#ifndef NDARRAY_H
#define NDARRAY_H

#include <initializer_list>
#include <type_traits>

#include <algorithm>
#include <array>
#include <concepts>
#include <stdexcept>

template< typename, size_t >
class NDArray;

template< typename, size_t >
class NDArrayViewBase;

namespace details
{
	template< typename U, size_t M >
	struct NestedInitializerList
	{
		using type = std::initializer_list< typename NestedInitializerList< U, M - 1 >::type >;
	};

	template< typename U >
	struct NestedInitializerList< U, 0 >
	{
		using type = U;
	};
	template< typename U, typename T >
	concept HasTotalCount = requires(const U& u) {
		{ u.total_count() } -> std::convertible_to< size_t >;
		{ u.data() } -> std::convertible_to< const T* >;
	};
	template< size_t N >
	void check_index(size_t index, const std::array< size_t, N >& dims)
	{
		if (index >= dims[0])
			throw std::out_of_range("Index out of range");
	}
	template< size_t N >
	size_t valid_index(const std::initializer_list< size_t >& idxs, const std::array< size_t, N >& dims)
	{
		if (idxs.size() != 1)
			throw std::invalid_argument("Requires exactly one index");
		size_t i = *idxs.begin();
		check_index(i, dims);
		return i;
	}
	template< size_t N >
	inline size_t all_size(const std::array< size_t, N >& dims) noexcept
	{
		size_t total = 1;
		for (size_t i = 0; i < N; ++i)
			total *= dims[i];
		return total;
	}
	using diff = std::ptrdiff_t;
	template< size_t N >
	inline diff difference(std::ptrdiff_t n, const std::array< size_t, N >& dims) noexcept
	{
		return n * static_cast< diff >(all_size(dims) / dims[0]);
	}
	template< size_t N >
	inline diff step_size(const std::array< size_t, N >& dims) noexcept
	{
		return all_size(dims) / dims[0];
	}
}	 // namespace details

// Это БАЗА
template< typename T, size_t N >
class NDBase
{
  public:
	using size_type = size_t;
	using difference_type = std::ptrdiff_t;
	using value_type = T;
	using pointer = T*;
	using reference = std::conditional_t< std::is_const_v< value_type >, const T&, T& >;

	NDBase() noexcept : data_(nullptr) {}

	size_type count() const noexcept { return dims_[0]; }

	size_type total_count() const noexcept { return details::all_size(dims_); }

	static inline size_type dim() noexcept { return N; }
	T* data() noexcept { return data_; }
	const T* data() const noexcept { return data_; }

	bool is_equal(const NDBase& rhs) const noexcept
		requires std::equality_comparable< T >
	{
		if (dims_ != rhs.dims_)
			return false;
		size_t total = total_count();
		for (size_t i = 0; i < total; ++i)
			if (!(data_[i] == rhs.data_[i]))
				return false;
		return true;
	}

	template< typename Derived, typename SizeList >
		requires std::is_arithmetic_v< typename SizeList::value_type > && (N > 0)
	Derived reshape(const SizeList& new_sizes) const
	{
		std::array< size_t, N > new_dims{};
		size_t i = 0;
		for (const typename SizeList::value_type& s : new_sizes)
		{
			if (i >= N)
				throw std::invalid_argument("Too many dims");
			new_dims[i++] = s;
		}
		if (i != N)
			throw std::invalid_argument("Not enough dims");
		size_t new_total = details::all_size(new_dims);
		if (new_total != total_count())
			throw std::invalid_argument("Size mismatch");
		std::array< size_t, N > new_strides{};
		new_strides[N - 1] = 1;
		for (difference_type k = static_cast< difference_type >(N) - 2; k >= 0; --k)
			new_strides[k] = new_strides[k + 1] * new_dims[k + 1];
		return Derived(data_, new_dims, new_strides);
	}

	void swap(NDBase& other) noexcept
	{
		std::swap(data_, other.data_);
		std::swap(dims_, other.dims_);
		std::swap(strides_, other.strides_);
	}

	template< bool Flag >
	NDArrayViewBase< std::conditional_t< Flag, const T, T >, N - 1 > get_subview(size_type i) const
	{
		using ptr_type = std::conditional_t< Flag, const T*, T* >;
		using view_type = NDArrayViewBase< std::conditional_t< Flag, const T, T >, N - 1 >;
		details::check_index(i, dims_);
		ptr_type sub_data = this->data_ + i * this->strides_[0];
		std::array< size_t, N - 1 > sub_dims;
		std::array< size_t, N - 1 > sub_strides;
		for (size_t k = 0; k < N - 1; ++k)
		{
			sub_dims[k] = this->dims_[k + 1];
			sub_strides[k] = this->strides_[k + 1];
		}
		return view_type(sub_data, sub_dims, sub_strides);
	}

	NDArrayViewBase< const T, N - 1 > at(const std::initializer_list< size_t >& indexes) const
		requires(N > 1)
	{
		size_t i = details::valid_index(indexes, dims_);
		return get_subview< true >(i);
	}

	const T& at(const std::initializer_list< size_t >& indexes) const
		requires(N == 1)
	{
		size_t i = details::valid_index(indexes, dims_);
		return data_[i];
	}

	NDArrayViewBase< T, N - 1 > at(const std::initializer_list< size_t >& indexes)
		requires(N > 1)
	{
		size_t i = details::valid_index(indexes, dims_);
		return get_subview< false >(i);
	}

	NDArrayViewBase< T, N - 1 > operator[](size_type i)
		requires(N > 1)
	{
		return get_subview< false >(i);
	}

	NDArrayViewBase< const T, N - 1 > operator[](size_type i) const
		requires(N > 1)
	{
		return get_subview< true >(i);
	}

	reference operator[](size_type i) const noexcept
		requires(N == 1)
	{
		return data_[i];
	}

	NDArrayViewBase< T, N > begin() noexcept
	{
		return NDArrayViewBase< T, N >(this->data_, this->dims_, this->strides_);
	}
	NDArrayViewBase< T, N > end() noexcept
	{
		T* ptr = this->data_ + this->total_count();
		return NDArrayViewBase< T, N >(ptr, std::array< size_t, N >{}, std::array< size_t, N >{});
	}

	// Константные версии
	NDArrayViewBase< const T, N > begin() const noexcept
	{
		return NDArrayViewBase< const T, N >(this->data_, this->dims_, this->strides_);
	}
	NDArrayViewBase< const T, N > end() const noexcept
	{
		const T* ptr = this->data_ + this->total_count();
		return NDArrayViewBase< const T, N >(ptr, std::array< size_t, N >{}, std::array< size_t, N >{});
	}
	NDArrayViewBase< const T, N > cbegin() const noexcept { return begin(); }
	NDArrayViewBase< const T, N > cend() const noexcept { return end(); }

  protected:
	std::array< size_t, N > dims_{};
	std::array< size_t, N > strides_{};
	pointer data_;

	void calc_strides() noexcept
	{
		strides_[N - 1] = 1;
		for (difference_type i = static_cast< difference_type >(N) - 2; i >= 0; --i)
			strides_[i] = strides_[i + 1] * dims_[i + 1];
	}
};

// Базовый класс для View
template< typename T, size_t N >
class NDArrayViewBase : public NDBase< T, N >
{
  public:
	using Base = NDBase< T, N >;
	using size_type = typename Base::size_type;
	using difference_type = typename Base::difference_type;
	using pointer = typename Base::pointer;
	using reference = typename Base::reference;
	using this_type = NDArrayViewBase;

	// Указатель на данные и размеры
	NDArrayViewBase(pointer data, const std::array< size_t, N >& dims_src, const std::array< size_t, N >& strides_src) noexcept
	{
		this->data_ = data;
		this->dims_ = dims_src;
		this->strides_ = strides_src;
	}
	// От двух итераторов(так как у меня уже вью я могу так делать)
	NDArrayViewBase(const this_type& begin, const this_type& end) noexcept
	{
		this->dims_ = begin.dims_;
		this->strides_ = begin.strides_;
		difference_type diff = end.data_ - begin.data_;
		if constexpr (N > 1)
			this->dims_[0] = diff / this->strides_[0];
		else
			this->dims_[0] = diff;
		this->data_ = begin.data_;
	}
	// От NDArray
	template< typename U >
	explicit NDArrayViewBase(const NDArray< U, N >& arr) noexcept
		requires std::is_same_v< T, const U > || std::is_same_v< T, U >
	{
		this->data_ = arr.data_;
		this->dims_ = arr.dims_;
		this->strides_ = arr.strides_;
	}
	// Конструторы из правила пяти
	NDArrayViewBase(const this_type&) noexcept = default;
	NDArrayViewBase(this_type&&) noexcept = default;
	NDArrayViewBase& operator=(const this_type&) noexcept = default;
	NDArrayViewBase& operator=(this_type&&) noexcept = default;

	// Операторы
	NDArrayViewBase< const T, N - 1 > operator*() const
		requires(N > 1)
	{
		return this->template get_subview< true >(0);
	}

	reference operator*() const noexcept
		requires(N == 1)
	{
		return *this->data_;
	}

	difference_type operator-(const this_type& o) const noexcept
	{
		return static_cast< difference_type >(this->data_ - o.data_);
	}

	this_type& operator++() noexcept
	{
		this->data_ += details::step_size(this->dims_);
		return *this;
	}

	this_type operator++(int) noexcept
	{
		this_type tmp = *this;
		++(*this);
		return tmp;
	}

	this_type& operator--() noexcept
	{
		this->data_ -= details::step_size(this->dims_);
		return *this;
	}

	this_type operator--(int) noexcept
	{
		this_type tmp = *this;
		--(*this);
		return tmp;
	}

	this_type operator+(difference_type n) const noexcept
	{
		this_type tmp = *this;
		tmp.data_ += details::difference(n, this->dims_);
		return tmp;
	}

	this_type operator-(difference_type n) const noexcept
	{
		this_type tmp = *this;
		tmp.data_ -= details::difference(n, this->dims_);
		return tmp;
	}

	this_type& operator+=(difference_type n) noexcept
	{
		this->data_ += details::difference(n, this->dims_);
		return *this;
	}

	this_type& operator-=(difference_type n) noexcept
	{
		this->data_ -= details::difference(n, this->dims_);
		return *this;
	}

	this_type* operator->() noexcept { return this; }

	const this_type* operator->() const noexcept { return this; }

	bool operator==(const this_type& o) const noexcept { return this->data_ == o.data_; }
	bool operator!=(const this_type& o) const noexcept { return this->data_ != o.data_; }
	bool operator<(const this_type& o) const noexcept { return this->data_ < o.data_; }
	bool operator>(const this_type& o) const noexcept { return this->data_ > o.data_; }
	bool operator<=(const this_type& o) const noexcept { return this->data_ <= o.data_; }
	bool operator>=(const this_type& o) const noexcept { return this->data_ >= o.data_; }

	this_type reshape(const std::initializer_list< size_t >& new_sizes) const
	{
		return Base::template reshape< this_type >(new_sizes);
	}
};

template< typename T, size_t N >
using NDArrayView = NDArrayViewBase< T, N >;
template< typename T, size_t N >
using NDArrayConstView = NDArrayViewBase< const T, N >;

// Класс NDArray
template< typename T, size_t N >
class NDArray : public NDBase< T, N >
{
  public:
	using Base = NDBase< T, N >;
	using size_type = typename Base::size_type;
	using pointer = T*;
	using view = NDArrayViewBase< T, N >;
	using const_view = NDArrayViewBase< const T, N >;
	using NestedInitList = details::NestedInitializerList< T, N >;

  private:
	void update_dims(const size_type* dims_ptr)
	{
		for (size_t i = 0; i < N; ++i)
			this->dims_[i] = dims_ptr[i];
		this->calc_strides();
	}

	// Выделение памяти с обработкой исключений
	template< typename Func >
	void mem_allocation(size_t total, Func&& func)
	{
		this->data_ = static_cast< T* >(operator new(sizeof(T) * total));
		size_t i = 0;
		try
		{
			for (i = 0; i < total; ++i)
				func(i);
		} catch (...)
		{
			clean(i);
			throw;
		}
	}

	// Заполнения
	void init_def()
		requires std::default_initializable< T >
	{
		size_t total = this->total_count();
		mem_allocation(total, [this](size_t i) { new (this->data_ + i) T(); });
	}

	void init_fill(const T& value)
	{
		size_t total = this->total_count();
		mem_allocation(total, [this, &value](size_t i) { new (this->data_ + i) T(value); });
	}

	void init_mem(const T* memory)
	{
		size_t total = this->total_count();
		mem_allocation(total, [this, memory](size_t i) { new (this->data_ + i) T(memory[i]); });
	}

	void add_data(const T& item, size_t& index) { new (this->data_ + index++) T(item); }

	template< typename U >
	void update_max_dims(const std::initializer_list< U >& init, size_t depth)
	{
		this->dims_[depth] = std::max(this->dims_[depth], init.size());
		if (depth + 1 < N)
		{
			for (const U& sub : init)
				update_max_dims(sub, depth + 1);
		}
	}

	void update_max_dims(const std::initializer_list< T >& init, size_t depth)
	{
		this->dims_[depth] = std::max(this->dims_[depth], init.size());
	}

	template< typename U >
	// Запись в многомерный массив для init-листов
	void fill_from_init_list(const std::initializer_list< U >& init, size_t& idx, size_t depth)
	{
		for (const U& sub : init)
		{
			fill_from_init_list(sub, idx, depth + 1);
		}
	}

	void fill_from_init_list(const std::initializer_list< T >& init, size_t& idx, size_t depth)
		requires std::default_initializable< T >
	{
		size_t count = 0;
		for (const T& val : init)
		{
			new (this->data_ + idx++) T(val);
			++count;
		}
		while (count < this->dims_[depth])
		{
			new (this->data_ + idx++) T();
			++count;
		}
	}

	// Освобождение памяти
	void remove() noexcept
	{
		if (this->data_)
		{
			size_t count = this->total_count();
			clean(count);
		}
	}

	void clean(size_t index) noexcept
	{
		for (size_t j = 0; j < index; ++j)
			(this->data_ + j)->~T();
		operator delete(this->data_);
		this->data_ = nullptr;
	}

	template< typename View >
	void construct_from_view(const View& v)
	{
		this->dims_ = v.dims_;
		this->strides_ = v.strides_;
		size_t total = this->total_count();
		mem_allocation(total, [this, &v](size_t i) { new (this->data_ + i) T(v.data_[i]); });
	}

  public:
	NDArray() noexcept
	{
		this->data_ = nullptr;
		this->strides_.fill(0);
		this->dims_.fill(0);
	}

	// Конструктор принимающий размеры каждой размерности и элемент, которым надо заполнить объект.
	NDArray(const std::array< size_t, N >& dims_src, const T& value)
		requires(N > 0)
	{
		this->dims_ = dims_src;
		init_fill(value);
		this->calc_strides();
	}

	// Конструктор принимающий размеры каждой размерности и указатель на начало области памяти
	NDArray(const std::array< size_t, N >& dims_src, const T* memory)
		requires(N > 0)
	{
		this->dims_ = dims_src;
		init_mem(memory);
		this->calc_strides();
	}

	// Конструктор принимающий размеры каждой размерности.
	template< typename... Sizes >
	explicit NDArray(Sizes... sizes) noexcept
		requires(sizeof...(Sizes) == N) && (std::integral< Sizes > && ...)
	{
		size_t tmp_dims[N] = { static_cast< size_type >(sizes)... };
		update_dims(tmp_dims);
		init_def();
	}

	// Конструкторы от view и const_view
	explicit NDArray(const view& v) { construct_from_view(v); }

	explicit NDArray(const const_view& v) { construct_from_view(v); }

	// Конструктор из диапазона итераторов
	template< std::input_iterator Iterator >
	NDArray(Iterator begin, Iterator end)
	{
		size_t arr = 0;
		for (auto it = begin; it != end; ++it)
			arr++;
		this->dims_[0] = arr;
		if constexpr (N > 1)
		{
			for (auto it = begin; it != end; ++it)
				update_max_dims(*it, 1);
		}
		this->calc_strides();
		size_type total = this->total_count();
		size_type element_count = 0;
		for (auto it = begin; it != end; ++it)
		{
			if constexpr (std::is_same_v< std::remove_reference_t< decltype(*it) >, T >)
				element_count += 1;
			else
				element_count += it->total_count();
		}
		if (element_count != total)
			throw std::invalid_argument("Different count");
		this->data_ = static_cast< T* >(operator new(sizeof(T) * total));
		size_t idx = 0;
		try
		{
			for (auto it = begin; it != end; ++it)
				add_data(*it, idx);
		} catch (...)
		{
			clean(idx);
			throw;
		}
	}

	// Конструктор из вложенного initializer_list
	explicit NDArray(typename NestedInitList::type init)
		requires(N > 0)
	{
		this->dims_.fill(0);
		this->dims_[0] = init.size();
		if constexpr (N > 1)
		{
			for (const auto& sub : init)
				update_max_dims(sub, 1);
		}
		size_t total = this->total_count();
		this->data_ = static_cast< T* >(operator new(sizeof(T) * total));
		size_t idx = 0;
		try
		{
			fill_from_init_list(init, idx, 0);
		} catch (...)
		{
			clean(idx);
			throw;
		}
		this->calc_strides();
	}

	explicit NDArray(typename NestedInitList::type init)
		requires(N == 0)
	{
		init_fill(init);
	}

	// Конструктор копирования
	NDArray(const NDArray& other)
		requires(N > 0)
	{
		this->dims_ = other.dims_;
		this->strides_ = other.strides_;
		size_t total = this->total_count();
		mem_allocation(total, [this, &other](size_t i) { new (this->data_ + i) T(other.data_[i]); });
	}

	NDArray(const NDArray& other)
		requires(N == 0)
	{
		this->data_ = static_cast< T* >(operator new(sizeof(T)));
		new (this->data_) T(*other.data_);
	}

	// Конструктор перемещения
	NDArray(NDArray&& other) noexcept
		requires(N > 0)
	{
		this->data_ = other.data_;
		this->dims_ = other.dims_;
		this->strides_ = other.strides_;
		other.dims_.fill(0);
		other.strides_.fill(0);
		other.data_ = nullptr;
	}

	NDArray(NDArray&& other) noexcept
		requires(N == 0)
	{
		this->data_ = other.data_;
		other.data_ = nullptr;
	}

	// Оператор копирующего присваивания
	NDArray& operator=(const NDArray& other)
	{
		if (this != &other)
			NDArray(other).swap(*this);
		return *this;
	}

	// Оператор перемещающего присваивания
	NDArray& operator=(NDArray&& other) noexcept
	{
		if (this != &other)
		{
			this->swap(other);
			other.remove();
		}
		return *this;
	}

	~NDArray() { remove(); }
	// Reshape для конст и не для конст сразу
	template< typename View >
	View reshape(const std::initializer_list< size_t >& new_sizes) const
	{
		return NDBase< T, N >::template reshape< View >(new_sizes);
	}
};
#endif	  // NDARRAY_H