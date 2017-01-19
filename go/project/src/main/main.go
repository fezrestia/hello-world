package main

import (
	f "fmt"
	_ "gosample"
	_ "strings"
    "errors"
    "log"
)

var message string = "var sample"

var multiline string = `line 1
line 2
line 3`

var hoge, fuga, piyo string = "hoge", "fuga", "piyo"

func main() {
//	f.Println(gosample.Message)
	f.Println(message)
	f.Println(multiline)

    var a = 10
    var b = 100

    if a < b {
        f.Println("b is larger than a")
    } else if a == b {
        f.Println("a equals b")
    } else {
        f.Println("a is larger than b")
    }

    for i := 0; i < 5; i++ {
        f.Println(i)
    }

    n := 10
    switch n {
        case 10:
            f.Println("case 10")

            fallthrough
        case 20:
            f.Println("case 20")
        case 30:
            f.Println("case 30")
        default:
            f.Println("default")
    }

    f.Println("10 + 5 =")
    f.Println(sum(10, 5))

    f.Println("swap 100, 200")
    x, y := 100, 200
    x, y = swap(x, y)
    f.Println(x)
    f.Println(y)

    f.Println("100 / 10 =")
    d, err := div(100, 10)
    if err != nil {
        log.Fatal(err)
    }
    f.Println(d)


    var array [4]string
    array[0] = "a"
    array[1] = "b"
    array[2] = "c"
    array[3] = "d"
    f.Println(array[0])
    array1 := [4]string{"a", "b", "c", "d"}
    f.Println(array1[0])
    array2 := [...]string{"a", "b", "c", "d"}
    f.Println(array2[0])


    var slice []string
    slice2 := []string{"aa", "bb", "cc", "dd"}
    f.Println(slice2[0])

    slice = append(slice, "a")
    slice = append(slice, "b", "c", "d")
    f.Println(slice)


    var month map[int]string = map[int]string{
        3: "March",
        4: "April",
    }
    month[1] = "January"
    month[2] = "February"
    f.Println(month)

    var isExist bool = false
    _, isExist = month[1]
    if isExist {
        f.Println("January is exist")
    }

    delete(month, 1)
    _, isExist = month[1]
    if !isExist {
        f.Println("January is NOT exist")
    }



}

func sum(i int, j int) int {
    return i + j
}

func swap(i int, j int) (int, int) {
    return j, i
}

func div(i int, j int) (result int, err error) {
    if j == 0 {
        err = errors.New("divided by zero")
        return // = return 0, err
    }

    result = i / j
    return // = return result, nil
}


