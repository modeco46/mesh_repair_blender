# Вычислительное ядро

Каталог `mesh_repair_blender/core` перенесён из **MeshRepair 1.0.0**, commit
[`6ca780f`](https://github.com/modeco46/MeshRepair/commit/6ca780f), без изменения алгоритмов ремонта.

Не перенесены Qt/VTK GUI, отдельный viewport, темы, виджеты и код запуска desktop-приложения. Blender отвечает за
сцену, отображение и сохранение `.blend`; `worker.py` сохраняет изоляцию вычислений, прогресс и остановку процесса.

При обновлении ядра нужно синхронно перенести `compute.py`, `deviation.py`, `engine.py`, `holes.py`, `inspection.py`,
`orientation.py`, `planes.py`, `reduction.py`, `reporting.py`, `storage.py`, обновить commit выше и выполнить тесты.
