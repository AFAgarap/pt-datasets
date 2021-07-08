# PyTorch Datasets utility repository
# Copyright (C) 2020  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Dataset classes"""
from pt_datasets.datasets.AGNews import AGNews
from pt_datasets.datasets.COVID19Dataset import (
    BinaryCOVID19Dataset,
    MultiCOVID19Dataset,
)
from pt_datasets.datasets.IMDB import IMDB
from pt_datasets.datasets.MalImg import MalImg
from pt_datasets.datasets.TwentyNewsgroups import TwentyNewsgroups
from pt_datasets.datasets.WDBC import WDBC
from pt_datasets.datasets.Yelp import Yelp
