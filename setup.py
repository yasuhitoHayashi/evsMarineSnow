from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext  # setuptools から build_ext を使用
import pybind11

ext_modules = [
    Extension(
        'particle_tracking',  # モジュール名
        ['particle_tracking.cpp'],  # コンパイルする C++ ファイル
        include_dirs=[pybind11.get_include()],  # pybind11 のヘッダファイルのディレクトリ
        extra_compile_args=['-O3', '-std=c++11'],  # C++11を有効にするオプションを追加
    ),
]

setup(
    name='particle_tracking',  # パッケージ名
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},  # setuptools の build_ext を指定
)