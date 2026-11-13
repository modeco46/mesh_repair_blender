# Сторонние компоненты

- **MeshLib / meshlib-core 3.1.4.297** — нативная обработка сеток, многопоточный CPU-код и CUDA bindings.
  [Репозиторий и условия](https://github.com/MeshInspector/MeshLib). Это source-available продукт; получение
  бинарников через pip не даёт дополнительных коммерческих прав. Для коммерческого распространения проверьте
  условия MeshInspector отдельно.
- **NumPy 2.2.4** — передача больших массивов между Blender и worker.
  [Лицензия BSD-3-Clause](https://numpy.org/doc/stable/license.html).
- **SciPy 1.16.3** — нативные разреженные графы согласования ориентации.
  [Лицензия BSD-3-Clause](https://github.com/scipy/scipy/blob/v1.16.3/LICENSE.txt).
- **Blender 5.3 / Python 3.13** — хост, UI, сцена и отображение результата.
  Blender распространяется по GPL; Python — по Python Software Foundation License.

Release ZIP включает бинарные пакеты MeshLib, NumPy и SciPy для конкретной платформы. При распространении
сборок необходимо сохранять уведомления и соблюдать условия всех поставляемых компонентов.
