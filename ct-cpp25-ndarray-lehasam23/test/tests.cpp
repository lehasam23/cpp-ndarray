#include "NDArray.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <ostream>

TEST(NDArrayTest, Construction)
{
	int arr1p[] = { 1, 2, 3 };
	NDArray< int, 1 > arr1d{ 1, 2, 3 };
	EXPECT_EQ(arr1d.count(), 3);
	EXPECT_EQ(arr1d.data()[0], 1);
	EXPECT_EQ(arr1d.data()[1], 2);
	EXPECT_EQ(arr1d.data()[2], 3);

	int arr2p[] = { 1, 2, 3, 4 };
	NDArray< int, 2 > arr2d{ { 1, 2 }, { 3, 4 } };
	EXPECT_EQ(arr2d.count(), 2);
	for (size_t i = 0; i < sizeof(arr2p) / sizeof(*arr2p); ++i)
	{
		EXPECT_EQ(arr2d.data()[i], arr2p[i]);
	}

	int arr3p[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	NDArray< int, 3 > arr3d{ { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
	EXPECT_EQ(arr3d.count(), 2);
	for (size_t i = 0; i < sizeof(arr3p) / sizeof(*arr3p); ++i)
	{
		EXPECT_EQ(arr3d.data()[i], arr3p[i]);
	}
}

TEST(NDArrayTest, CopyAndMove)
{
	NDArray< int, 2 > original{ { 1, 2 }, { 3, 4 } };
	NDArray< int, 2 > copied(original);
	EXPECT_EQ(copied.count(), 2);
	EXPECT_EQ(copied.data()[0], 1);
	EXPECT_EQ(copied.data()[1], 2);
	EXPECT_EQ(copied.data()[2], 3);
	EXPECT_EQ(copied.data()[3], 4);

	NDArray< int, 2 > moved(std::move(original));
	EXPECT_EQ(moved.count(), 2);
	EXPECT_EQ(moved.data()[0], 1);
	EXPECT_EQ(moved.data()[1], 2);
	EXPECT_EQ(moved.data()[2], 3);
	EXPECT_EQ(moved.data()[3], 4);
	EXPECT_EQ(original.data(), nullptr);
}

TEST(NDArrayTest, CopyAndMoveTraits)
{
	EXPECT_TRUE((!std::is_trivially_copyable_v< NDArray< int, 2 > >))
		<< "Pointer-owning NDArray must not be trivially "
		   "copyable";

	EXPECT_TRUE((std::is_copy_constructible_v< NDArray< int, 2 > >)) << "Should be copy-constructible";

	EXPECT_TRUE((std::is_move_constructible_v< NDArray< int, 2 > >)) << "Should be move-constructible";

	EXPECT_TRUE((std::is_nothrow_move_constructible_v< NDArray< int, 2 > >))
		<< "Move-construct should be noexcept for "
		   "basic types";
}

TEST(NDArrayTest, DifferentTypes)
{
	NDArray< double, 2 > doubleArr{ { 1.5, 2.5 }, { 3.5, 4.5 } };
	EXPECT_EQ(doubleArr.count(), 2);
	EXPECT_DOUBLE_EQ(doubleArr.data()[0], 1.5);
	EXPECT_DOUBLE_EQ(doubleArr.data()[1], 2.5);
	EXPECT_DOUBLE_EQ(doubleArr.data()[2], 3.5);
	EXPECT_DOUBLE_EQ(doubleArr.data()[3], 4.5);
}

TEST(NDArrayTest, total_countMethod)
{
	NDArray< int, 1 > arr1d{ 1, 2, 3 };
	EXPECT_EQ(arr1d.total_count(), 3);

	NDArray< int, 2 > arr2d{ { 1, 2 }, { 3, 4 } };
	EXPECT_EQ(arr2d.total_count(), 4);

	NDArray< int, 3 > arr3d{ { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
	EXPECT_EQ(arr3d.total_count(), 8);

	NDArray< int, 4 > arr4d{ { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, { { { 9, 10 }, { 11, 12 } }, { { 13, 14 }, { 15, 16 } } } };
	EXPECT_EQ(arr4d.total_count(), 16);
}

TEST(NDArrayTest, EmptyArray)
{
	NDArray< int, 1 > empty1d{};
	EXPECT_EQ(empty1d.count(), 0);
	EXPECT_EQ(empty1d.total_count(), 0);
	EXPECT_EQ(empty1d.data(), nullptr);

	NDArray< int, 2 > empty2d{ {} };
	EXPECT_EQ(empty2d.count(), 1);
	EXPECT_EQ(empty2d.total_count(), 0);
	EXPECT_NE(empty2d.data(), nullptr);
}

TEST(NDArrayMoveAssignment, BasicMoveAssignmentStealsStorage)
{
	NDArray< int, 2 > src{ { 1, 2 }, { 3, 4 } };
	int* oldPtr = src.data();

	NDArray< int, 2 > dst{ { 1, 1 }, { 1, 1 } };
	dst = std::move(src);

	EXPECT_EQ(dst.data()[0], 1);
	EXPECT_EQ(dst.data()[3], 4);
	EXPECT_EQ(src.data(), nullptr);
	EXPECT_EQ(dst.data(), oldPtr);
}

TEST(NDArrayMoveAssignment, SelfMoveNoCrashNoChange)
{
	NDArray< int, 1 > arr{ 10, 20, 30 };
	int* before = arr.data();
	size_t len = arr.total_count();

	arr = std::move(arr);
	EXPECT_EQ(arr.data(), before);
	EXPECT_EQ(arr.total_count(), len);
}

struct LifeCounter
{
	static inline int live = 0;
	int v{};

	explicit LifeCounter(int v = 0) : v(v) { ++live; }
	LifeCounter(const LifeCounter& other) : v(other.v) { ++live; }
	LifeCounter(LifeCounter&& other) noexcept : v(other.v)
	{
		++live;
		other.v = -1;
	}

	LifeCounter& operator=(const LifeCounter& o)
	{
		v = o.v;
		return *this;
	}

	LifeCounter& operator=(LifeCounter&& o) noexcept
	{
		v = o.v;
		o.v = -1;
		return *this;
	}

	~LifeCounter() { --live; }
	bool operator==(const LifeCounter& o) const { return v == o.v; }
};

TEST(NDArrayLifetime, ConstructionsEqualDestructions)
{
	{
		NDArray< LifeCounter, 3 > arr{ { { LifeCounter(1), LifeCounter(2) } }, { { LifeCounter(3), LifeCounter(4) } } };
		EXPECT_EQ(LifeCounter::live, 4);
	}
	EXPECT_EQ(LifeCounter::live, 0);
}

TEST(NDArrayConstAccess, DataAndIndexingAreConstCorrect)
{
	const NDArray< std::string, 2 > names{ { std::string("A"), std::string("B") }, { std::string("C"), std::string("D") } };
	EXPECT_EQ(names[1][0], std::string("C"));
}

TEST(NDArrayConstAccess, DataAndIndexingAreConstCorrectTraits)
{
	const NDArray< std::string, 2 > names{ { std::string("A"), std::string("B") }, { std::string("C"), std::string("D") } };

	EXPECT_TRUE((std::is_same_v< decltype(names.data()), const std::string* >))
		<< "data() const should be "
		   "const-qualified";

	EXPECT_TRUE((std::is_same_v< decltype(names[1][0]), std::string const & >)) << "[] const should be const-qualified";

	NDArray< bool, 1 > arr{ true, false, true };
	EXPECT_TRUE((std::is_same_v< decltype(arr.data()), bool* >)) << "data() should not transform bool";
}

TEST(NDArrayComparisonEdge, DifferentShapesNeverEqual)
{
	NDArray< int, 1 > a{ 1, 2, 3 };
	NDArray< int, 1 > b{ 1, 2, 3, 4 };
	EXPECT_FALSE(a.is_equal(b));
}

TEST(NDArrayCopyIndependence, MutateCopyDoesNotAffectOriginal)
{
	NDArray< int, 2 > org{ { 1, 2 }, { 3, 4 } };
	NDArray< int, 2 > cpy(org);

	cpy[0][0] = 99;
	EXPECT_EQ(org[0][0], 1);
	EXPECT_EQ(cpy[0][0], 99);
}

TEST(NDArrayHighNension, SevenNOnes)
{
	NDArray< int, 7 > ones{ { { { { { { 1 } } } } } } };
	EXPECT_EQ(ones.total_count(), 1);
	ones[0][0][0][0][0][0][0] = 42;
	EXPECT_EQ(ones[0][0][0][0][0][0][0], 42);
}

TEST(NDArrayZeroNensional, CopyAndMoveOps)
{
	NDArray< int, 0 > v(7);
	NDArray< int, 0 > cpy = v;
	EXPECT_EQ(*cpy.data(), 7);

	NDArray< int, 0 > mv = std::move(cpy);
	EXPECT_EQ(*mv.data(), 7);
	EXPECT_EQ(cpy.data(), nullptr);
}

TEST(NDArrayBool, BasicOperations)
{
	NDArray< bool, 1 > arr{ true, false, true };

	EXPECT_EQ(arr.count(), 3u);
	EXPECT_TRUE(arr[0]);
	EXPECT_FALSE(arr[1]);
	EXPECT_TRUE(arr[2]);

	arr[1] = true;
	EXPECT_TRUE(arr[1]);
}

struct ThrowOnCopy
{
	inline static int copies = 0;
	inline static int max_copies = 2;
	int id;

	explicit ThrowOnCopy(int id = 0) : id(id) {}

	ThrowOnCopy(const ThrowOnCopy& other) : id(other.id)
	{
		if (++copies > max_copies)
			throw std::runtime_error("copy failed");
	}

	ThrowOnCopy(ThrowOnCopy&&) noexcept = default;
	ThrowOnCopy& operator=(const ThrowOnCopy&) = default;
	ThrowOnCopy& operator=(ThrowOnCopy&&) noexcept = default;

	operator int() const { return id; }

	static void reset() { copies = 0; }

	static void set(int value) { max_copies = value; }

	bool operator==(const ThrowOnCopy& other) const { return id == other.id; }
};

TEST(NDArrayExceptionSafety, CopyCtorLeavesSourceIntactOnThrow)
{
	ThrowOnCopy::reset();
	NDArray< ThrowOnCopy, 1 > good{ ThrowOnCopy(1), ThrowOnCopy(2) };

	EXPECT_THROW(
		(
			[&]
			{
				NDArray< ThrowOnCopy, 1 > doomed(good);
				(void)doomed;
			}()),
		std::runtime_error);

	EXPECT_EQ(good.total_count(), 2u);
	EXPECT_EQ(good[0], ThrowOnCopy(1));
	EXPECT_EQ(good[1], ThrowOnCopy(2));
}

TEST(NDArraySwap, StdSwapsEverything)
{
	NDArray< int, 2 > a{ { 1, 2 }, { 3, 4 } };
	NDArray< int, 2 > b{ { 9, 8 }, { 7, 6 } };

	int* aPtr = a.data();
	int* bPtr = b.data();

	std::swap(a, b);

	EXPECT_EQ(a.data(), bPtr);
	EXPECT_EQ(b.data(), aPtr);
	EXPECT_EQ(a[0][0], 9);
	EXPECT_EQ(b[1][1], 4);
}

TEST(NDArraySwap, SwapsEverything)
{
	NDArray< int, 2 > a{ { 1, 2 }, { 3, 4 } };
	NDArray< int, 2 > b{ { 9, 8 }, { 7, 6 } };

	int* aPtr = a.data();
	int* bPtr = b.data();

	a.swap(b);

	EXPECT_EQ(a.data(), bPtr);
	EXPECT_EQ(b.data(), aPtr);
	EXPECT_EQ(a[0][0], 9);
	EXPECT_EQ(b[1][1], 4);
}
TEST(NDArrayIterator, ForwardAccumulate)
{
	NDArray< int, 1 > arr{ 1, 2, 3, 4 };
	int sum = std::accumulate(arr.begin(), arr.end(), 0);
	EXPECT_EQ(sum, 10);
}

TEST(NDArrayIterator, MutateViaIterator)
{
	NDArray< int, 1 > arr{ 1, 2, 3, 4, 5 };
	for (auto it = arr.begin(); it != arr.end(); ++it)
		*it *= 2;

	EXPECT_EQ(arr[0], 2);
	EXPECT_EQ(arr[4], 10);
}

TEST(NDArrayIterator, ConstAndReverseIterators)
{
	const NDArray< int, 1 > arr{ 1, 2, 3 };
	std::vector< int > fwd(arr.cbegin(), arr.cend());
	EXPECT_EQ(fwd, (std::vector< int >{ 1, 2, 3 }));
}

TEST(NDArrayIterator, ConstAndReverseIteratorsTraits)
{
	const NDArray< int, 1 > arr{ 1, 2, 3 };
	EXPECT_TRUE((std::same_as< decltype(arr.cbegin()), decltype(arr.begin()) >));
}

TEST(NDArraySubview, RowViewIsLive)
{
	NDArray< int, 2 > arr{ { 10, 11 }, { 20, 21 } };
	auto row = arr[1];
	row[0] = 77;
	EXPECT_EQ(arr[1][0], 77);
}

int main(int argc, char* argv[])
{
	// // test 1
	// NDArray<int, 0> scalar(42); // Скаляр со значением 42
	// auto it = scalar.begin();   // NDArrayView<int, 0>, указывает на 42
	// auto end = scalar.end();    // NDArrayView<int, 0>, указывает за элемент
	// if (it != end) {
	// 	*it = 100;              // Изменяет скаляр на 100
	// }
	//
	// std::cout << "-------------------------\n";
	//
	// // test 2.1
	// NDArray<int, 1> arr(5);
	//
	// // Print elements (all default-initialized to 0)
	// for (size_t i = 0; i < arr.count(); ++i) {
	// 	std::cout << arr[i] << " "; // Output: 0 0 0 0 0
	// }
	// std::cout << "\n";
	//
	// // Verify dimensions
	// std::cout << "Count: " << arr.count() << "\n"; // Output: Count: 5
	// std::cout << "Total count: " << arr.total_count() << "\n"; // Output: Total count: 5
	// std::cout << "Dimensions: " << arr.dim() << "\n";
	//
	// std::cout << "-------------------------\n";
	//
	// //test 2.2
	// NDArray<double, 2> arr2(3, 4);
	//
	// // Print elements (all default-initialized to 0.0)
	// for (size_t i = 0; i < arr2.count(); ++i) {
	// 	for (size_t j = 0; j < arr2[i].count(); ++j) {
	// 		std::cout << arr2[i][j] << " "; // Output: 0 0 0 0 0 0 0 0 0 0 0 0
	// 	}
	// 	std::cout << "\n";
	// }
	//
	// // Verify dimensions
	// std::cout << "Count (dim 0): " << arr2.count() << "\n"; // Output: Count (dim 0): 3
	// std::cout << "Total count: " << arr2.total_count() << "\n"; // Output: Total count: 12
	// std::cout << "Dimensions: " << arr2.dim() << "\n"; // Output: Dimensions: 2
	//
	// std::cout << "-------------------------\n";
	//
	// // test 2.3
	// NDArray<int, 1> arr3(0);
	// std::cout << "Count (dim 0): " << arr3.count() << "\n"; // Output: Count (dim 0): 3
	// std::cout << "Total count: " << arr3.total_count() << "\n"; // Output: Total count: 12
	// std::cout << "Dimensions: " << arr3.dim() << "\n";
	//
	// std::cout << "-------------------------\n";
	//
	// // test 3.1
	// NDArray<int, 1> arr4(5);
	//
	// // Заполняем массив значениями с помощью begin() и end()
	// int value = 1;
	// for (auto it = arr4.begin(); it != arr4.end(); ++it) {
	// 	*it = value++; // it имеет тип NDArrayView<int, 1>, *it дает int&
	// }
	//
	// // Выводим элементы с помощью cbegin() и cend() (константные итераторы)
	// std::cout << "Массив: ";
	// for (auto it = arr4.cbegin(); it != arr4.cend(); ++it) {
	// 	std::cout << *it << " "; // Вывод: 1 2 3 4 5
	// }
	// std::cout << "\n";
	//
	// // Используем стандартный алгоритм с итераторами
	// int sum = std::accumulate(arr4.begin(), arr4.end(), 0);
	// std::cout << "Сумма: " << sum << "\n";
	//
	// std::cout << "-------------------------\n";
	//
	// //test 3.2
	//
	// NDArray<int, 2> arr32{{1, 2}, {3, 4}, {5, 6}};
	// for (auto it = arr32.begin(); it != arr32.end(); ++it) {
	// 	for (size_t j = 0; j < (*it)->count(); ++j) { // Ошибка здесь
	// 		std::cout << (*it)[j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	//
	// std::cout << "-------------------------\n";
	//
	// // test 4
	//
	// NDArray<int, 2> arr_test4{{1, 2}, {3, 4}, {5, 6}}; // Массив 3x2
	// std::cout << "Количество строк (старшая размерность): " << arr_test4.count() << std::endl; // 3
	//
	// NDArray<int, 1> arr_test4_1{1, 2, 3}; // Массив 3
	// std::cout << "Количество элементов (старшая размерность): " << arr_test4_1.count() << std::endl; // 3
	//
	// std::cout << "-------------------------\n";
	// // test 5
	//
	// NDArray<int, 2> arr_test5{{1, 2}, {3, 4}, {5, 6}}; // Массив 3x2
	// std::cout << "Количество измерений: " << arr_test5.dim() << std::endl; // 2
	//
	// NDArray<int, 1> arr_test5_1{1, 2, 3}; // Массив 3
	// std::cout << "Количество измерений: " << arr_test5_1.dim() << std::endl; // 1
	//
	// NDArray<int, 0> arr_scalar(42); // Скаляр
	// std::cout << "Количество измерений: " << arr_scalar.dim() << std::endl; // 0
	//
	// std::cout << "-------------------------\n";
	// // test 6
	//
	// NDArray<int, 2> arr_test6{{1, 2}, {3, 4}, {5, 6}}; // Массив 3x2
	// auto row = arr_test6[1]; // Получаем NDArrayView<int, 1> для второй строки
	// std::cout << "Вторая строка: " << row << std::endl; // 3 4
	//
	// NDArray<int, 1> arr_test6_1{1, 2, 3}; // Массив 3
	// std::cout << "Элемент с индексом 1: " << arr_test6_1[1] << std::endl; // 2
	//
	//
	// std::cout << "-------------------------\n";
	// // test 7
	//
	//
	// NDArray<int, 2> arr_test7{{1, 2}, {3, 4}, {5, 6}}; // Массив 3x2
	// auto row_test7 = arr_test7.at({1}); // Получаем NDArrayView<int, 1> для второй строки
	// std::cout << "Вторая строка: " << row_test7 << std::endl; // 3 4
	//
	// NDArray<int, 1> arr_test7_1{1, 2, 3}; // Массив 3
	// std::cout << "Элемент с индексом 1: " << arr_test7_1.at({1}) << std::endl; // 2
	//
	// try {
	// 	arr_test7.at({5}); // Выход за границы
	// } catch (const std::out_of_range& e) {
	// 	std::cout << "Ошибка: " << e.what() << std::endl;
	// }
	//
	// std::cout << "-------------------------\n";
	// // test 8
	// NDArray<int, 2> arr_test8_1{{1, 2}, {3, 4}}; // Массив 2x2
	// NDArray<int, 2> arr_test8_2{{1, 2}, {3, 4}}; // Такой же массив
	// NDArray<int, 2> arr_test8_3{{1, 2}, {5, 6}}; // Другой массив
	//
	// std::cout << "arr1 == arr2: " << arr_test8_1.is_equal(arr_test8_2) << std::endl; // 1 (true)
	// std::cout << "arr1 == arr3: " << arr_test8_1.is_equal(arr_test8_3) << std::endl; // 0 (false)
	//
	// std::cout << "-------------------------\n";
	// //test 9
	//
	// NDArray<int, 2> arr_test9{{1, 2}, {3, 4}, {5, 6}}; // Массив 3x2
	//
	// // Использование begin() и end()
	// std::cout << "Проход с begin() и end():\n";
	// for (auto it = arr_test9.begin(); it != arr_test9.end(); ++it) {
	// 	std::cout << *it << std::endl; // Выводит строки: 1 2, 3 4, 5 6
	// }
	//
	// // Использование cbegin() и cend() для константного массива
	// const NDArray<int, 2> const_arr_test9{{1, 2}, {3, 4}, {5, 6}};
	// std::cout << "Проход с cbegin() и cend():\n";
	// for (auto it = const_arr_test9.cbegin(); it != const_arr_test9.cend(); ++it) {
	// 	std::cout << *it << std::endl; // Выводит строки: 1 2, 3 4, 5 6
	// }
	//
	// std::cout << "-------------------------\n";
	// //test 10
	//
	// NDArray<int, 2> arr_test10{{1, 2}, {3, 4}, {5, 6}}; // Массив 3x2 (6 элементов)
	// auto reshaped = arr_test10.reshape({2, 3}); // Переинтерпретируем как 2x3
	//
	// std::cout << "Массив после reshape (2x3):\n";
	// for (auto it = reshaped.begin(); it != reshaped.end(); ++it) {
	// 	std::cout << *it << std::endl; // Выводит строки: 1 2 3, 4 5 6
	// }
	//
	//
	// try {
	// 	arr_test10.reshape({4, 2}); // Ошибка: несоответствие размеров
	// } catch (const std::invalid_argument& e) {
	// 	std::cout << "Ошибка: " << e.what() << std::endl;
	// }
	//
	//
	// std::cout << "-------------------------\n";
	// //test 11
	//
	// NDArray<int, 2> arr_test11_1{{1, 2}, {3, 4}}; // Массив 2x2
	// NDArray<int, 2> arr_test11_2{{5, 6}, {7, 8}}; // Массив 2x2
	//
	// std::cout << "До swap:\narr1:\n";
	// for (auto it = arr_test11_1.begin(); it != arr_test11_1.end(); ++it) {
	// 	std::cout << *it << std::endl;
	// }
	// std::cout << "arr2:\n";
	// for (auto it = arr_test11_2.begin(); it != arr_test11_2.end(); ++it) {
	// 	std::cout << *it << std::endl;
	// }
	//
	// arr_test11_1.swap(arr_test11_2);
	//
	// std::cout << "После swap:\narr1:\n";
	// for (auto it = arr_test11_1.begin(); it != arr_test11_1.end(); ++it) {
	// 	std::cout << *it << std::endl;
	// }
	// std::cout << "arr2:\n";
	// for (auto it = arr_test11_2.begin(); it != arr_test11_2.end(); ++it) {
	// 	std::cout << *it << std::endl;
	// }
	//
	// std::cout << "-------------------------\n";
	// // test 12
	//
	// NDArray<int, 2> arr_test12{{1, 2}, {3, 4}, {5, 6}}; // Массив 3x2
	//
	// // Пример с итераторами
	// auto it1 = arr_test12.begin(); // NDArrayView<int, 2>
	// auto it2 = arr_test12.begin();
	//
	// // Сравнение
	// std::cout << "it1 == it2: " << (it1 == it2) << std::endl; // 1 (true)
	//
	// // Инкремент и декремент
	// ++it1; // Переходим к следующей строке
	// std::cout << "После ++it1: " << *it1 << std::endl; // 3 4
	// --it1; // Возвращаемся назад
	// std::cout << "После --it1: " << *it1 << std::endl; // 1 2
	//
	// // Сложение и вычитание
	// auto it3 = it1 + 2; // Переходим на 2 строки вперед
	// std::cout << "it1 + 2: " << *it3 << std::endl; // 5 6
	// auto it4 = it3 - 1; // На одну строку назад
	// std::cout << "it3 - 1: " << *it4 << std::endl; // 3 4
	//
	// // += и -=
	// it1 += 2;
	// std::cout << "После it1 += 2: " << *it1 << std::endl; // 5 6
	// it1 -= 1;
	// std::cout << "После it1 -= 1: " << *it1 << std::endl; // 3 4
	//
	// // Сравнение
	// std::cout << "it1 < arr.end(): " << (it1 < arr_test12.end()) << std::endl; // 1 (true)
	// std::cout << "it1 > it2: " << (it1 > it2) << std::endl; // 1 (true)
	//
	// // Оператор []
	// auto row_test12 = it1[0]; // NDArrayView<int, 1> для текущей строки
	// std::cout << "Текущая строка через it1[0]: " << row_test12 << std::endl; // 3 4
	//
	// // Оператор ->
	// std::cout << "Количество элементов в строке через ->: " << it1->count() << std::endl; // 2
	//
	// // Присваивание
	// auto it5 = arr_test12.begin();
	// it5 = it1;
	// std::cout << "После присваивания it5 = it1: " << *it5 << std::endl; // 3 4
	//
	// //test 13
	// std::cout << "-------------------------\n";
	//
	// NDArray<int, 2> matrix{{1, 2}, {3, 4}}; // 2x2 матрица
	//
	// // Выводим элементы для проверки
	// std::cout << "Matrix (2x2) elements:\n";
	// for (size_t i = 0; i < matrix.count(); ++i) {
	// 	std::cout << matrix[i] << std::endl; // Вывод строк
	// }
	//
	// //test 14
	// std::cout << "-------------------------\n";
	// NDArray<int, 2> matrix_test14(2, 3); // 2x3 матрица, инициализируется значениями по умолчанию (0)
	//
	// // Выводим элементы для проверки
	// std::cout << "Matrix (2x3) with default values:\n";
	// for (size_t i = 0; i < matrix_test14.count(); ++i) {
	// 	std::cout << matrix_test14[i] << std::endl; // Вывод строк (все 0)
	// }
	//
	//
	// // 2. Используя initializer list
	// NDArray<int, 2> matrix1{{1, 2}, {3, 4, 5, 6}, {7}}; // 2x2 матрица с инициализацией
	// std::cout << "Matrix1 (2x2) from initializer list:\n" << matrix1[0] << "\n" << matrix1[1] << "\n" << matrix1[2] << std::endl;
	//
	// // 3. Принимающий размеры
	// NDArray<int, 2> matrix2(2, 2);
	// std::cout << "Matrix2 (2x2) with default values:\n" << matrix2[0] << "\n" << matrix2[1] << "\n";
	//
	// // 4а. Размеры и указатель на данные
	// int data[] = {1, 2, 3, 4, 5, 6};
	// NDArray<int, 2> matrix3({2, 3}, data);
	// std::cout << "Matrix3 (2x3) from array:\n" << matrix3[0] << "\n" << matrix3[1] << "\n";
	//
	// // 4б. Указатель на область памяти
	// size_t dims[] = {2, 3};
	//
	//
	// // 5. От NDArrayView
	// NDArrayView<int, 2> view = matrix3.begin();
	// NDArray<int, 2> matrix6(view);
	// std::cout << "Matrix6 (2x3) from view:\n" << matrix6[0] << "\n" << matrix6[1] << "\n";
	//
	// // // 6. От двух итераторов
	// // auto it_begin = matrix3.begin();
	// // auto it_end = matrix3.begin() + 2; // Две строки
	// // NDArray<int, 2> matrix7(dims, it_begin, it_end);
	// // std::cout << "Matrix7 (2x3) from iterators:\n" << matrix7[0] << "\n" << matrix7[1] << "\n";
	// //
	// // NDArray<int, 2> matrix_new({ {1}, {2, 3} });
	// // for (size_t i = 0; i < matrix_new.count(); ++i) {
	// // 	std::cout << matrix_new[i] << std::endl; // Вывод строк (все 0)
	// // }
	// //
	// // size_t dims_col[2] = {2, 3};  // матрица 2x3
	// // std::vector<int> vec = {1, 2, 3, 4, 5, 6};
	// // NDArray<int, 2> arrFlat(dims_col, vec.begin(), vec.end());
	//
	// // std::cout << "First row:";
	// // for (size_t i = 0; i < 3; ++i)
	// // 	std::cout << " " << arrFlat[0][i];
	// // std::cout << "\nSecond row:";
	// // for (size_t i = 0; i < 3; ++i)
	// // 	std::cout << " " << arrFlat[1][i];
	// // std::cout << std::endl;
	//
	// // test
	//
	// // Пусть matrix имеет тип NDArray<int, 2>
	// NDArray<int, 2> matrix_finish(2, 3);  // создаём 2x3 матрицу
	// // Заполним matrix
	// for (size_t i = 0; i < matrix_finish.total_count(); ++i)
	// 	matrix_finish.data()[i] = static_cast<int>(i + 1);
	//
	// // Теперь создадим диапазон итераторов, копирующий первые две строки
	// // auto rowBegin = matrix_finish.begin();
	// // auto rowEnd = matrix_finish.begin() + 2;  // предположим, что такой сдвиг возможен
	// // size_t dims2[2] = {2, 3};
	// // NDArray<int, 2> arrNested(dims2, rowBegin, rowEnd);
	// //
	// // // Выводим строки нового массива
	// // std::cout << "New matrix from nested iterators:\n";
	// // for (size_t i = 0; i < 2; ++i) {
	// // 	for (size_t j = 0; j < 3; ++j)
	// // 		std::cout << arrNested[i][j] << " ";
	// // 	std::cout << "\n";
	// // }
	// NDArray<int, 0> scalar_new(42); // Инициализируем значением 42
	//
	// // Используем итератор
	// auto it_scalar = scalar_new.begin();
	// std::cout << "Разыменование итератора для N=0: " << *it << "\n";
	//
	// // Проверяем конец итератора
	// auto end_it = scalar_new.end();
	// std::cout << "Разница между началом и концом: " << (end_it - it_scalar) << "\n";
	//
	// // Create an initial NDArray
	// size_t dims44[] = {2, 2};
	// NDArray<int, 2> arr44(2, 2); // 2x2 array
	//
	// // Create a view
	// NDArrayView<int, 2> view1(arr44);
	// view1[0][0] = 10; // Modify through view1
	//
	// // Copy construct a new view
	// NDArrayView<int, 2> view2(view1);
	// std::cout << "view2[0] = " << view2[0] << std::endl; // Outputs 10
	//
	// // Move construct a new view
	// NDArrayView<int, 2> view3(std::move(view2));
	// std::cout << "view3[0] = " << view3[0] << std::endl; // Outputs 10
	//
	//
	// NDArray<int, 1> arr234456(3);
	// NDArrayView<int, 1> view4(arr234456);
	// std::cout << "view4[1] = " << view4[1] << std::endl;
	// view4[1] = 20;
	// std::cout << "Modified view4[1] = " << view4[1] << std::endl;
	//
	// NDArrayConstView<int, 1> constView4(arr234456);
	// std::cout << "constView4[2] = " << constView4[2] << std::endl;
	// NDArray<int, 2> arr_test_kanye({{2, 3}, {1, 2, 3}});
	//
	// // Создаем представление
	// NDArrayView<int, 2> view_kanye(arr_test_kanye);
	//
	// // Изменяем форму на 3x2
	// auto reshaped_view = view_kanye.reshape({3, 2});
	//
	// // Выводим элементы исходного массива
	// std::cout << "Исходный массив (2x3):\n";
	// for (size_t i = 0; i < 2; ++i) {
	// 	for (size_t j = 0; j < 3; ++j) {
	// 		std::cout << arr_test_kanye[i][j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	//
	// // Выводим элементы reshaped_view
	// std::cout << "\nПосле reshape (3x2):\n";
	// for (size_t i = 0; i < 3; ++i) {
	// 	for (size_t j = 0; j < 2; ++j) {
	// 		std::cout << reshaped_view[i][j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	//
	// std::vector<std::vector<int>> data2321 = {{1, 2}, {3, 4}};
	// // Создаём NDArray из initializer_list
	// NDArray<int, 2> arr3232{{1, 2}, {3, 4}};
	//
	// // Выводим данные через индексацию
	// std::cout << "NDArray elements:\n";
	// std::cout << arr3232[0][0] << " " << arr3232[0][1] << "\n"; // 1 2
	// std::cout << arr3232[1][0] << " " << arr3232[1][1] << "\n"; // 3 4
	// NDArrayViewBase<int, 2> view232(arr3232.begin(), arr3232.end());
	//
	// // Выводим данные через view
	// std::cout << "NDArrayBaseView elements:\n";
	// std::cout << view232[0][0] << " " << view232[0][1] << "\n"; // 1 2
	// std::cout << view232[1][0] << " " << view232[1][1] << "\n"; // 3 4
	//
	// NDArray<int, 0> a0(5);
	// NDArray<int, 0> b0(5);
	// assert(a0.is_equal(b0));               // same value
	// NDArray<int, 0> c0(7);
	// assert(!a0.is_equal(c0));
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
