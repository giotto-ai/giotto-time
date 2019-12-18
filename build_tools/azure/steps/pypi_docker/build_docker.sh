#!/bin/bash
docker run -d -i -t -e python_ver=$PYTHON_VER --name manylinux10 -v `pwd`:/io quay.io/pypa/manylinux2010_x86_64 /bin/bash
docker exec manylinux10 sh -c "sh /io/build_tools/azure/steps/pypi_docker/build_wheels_pypi.sh"