
cat requirements.txt | xargs pip install -U  # installs all packages in requirements.txt in order
echo "backend: TkAgg"> ~/.matplotlib/matplotlibrc  # fix matplotlib install issue

