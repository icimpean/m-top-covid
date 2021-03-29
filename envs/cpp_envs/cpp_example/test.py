# noinspection PyUnresolvedReferences
# import envs.cpp_envs.cpp_example.example as example
import example


if __name__ == '__main__':
    # Build command:
    # c++ -O3 -Wall -shared -std=c++11 $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
    #
    # -undefined dynamic_lookup added for macOS:
    # c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
    x = example.add(2, 8) + example.add(4, 5)
    print(x)

    help(example)
