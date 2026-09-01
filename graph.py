import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham


city_fp = os.path.join(proj_dir, 'data/city.txt')
altitude_fp = os.path.join(proj_dir, 'data/altitude.npy')


class Graph():
    def __init__(self, city_fp=city_fp, use_altitude=True, k_neighbors=None):
        """
        city_fp: path to a "idx city lon lat" file. Defaults to the
            original 184-city China list.
        use_altitude: whether to load data/altitude.npy and use it both as
            a node attribute and to prune edges that cross a ridge (see
            _update_edges). Only meaningful for the default China city_fp
            - altitude.npy is a raster over a fixed China bounding box
            (see _lonlat2xy), so it isn't valid for other regions. Pass
            False for any other city_fp; node_attr's altitude column comes
            back all-zero in that case.
        k_neighbors: if set, edges are built as a symmetric k-nearest-
            -neighbor graph by geodesic distance (see _gen_edges_knn)
            instead of the default fixed-degree distance threshold
            (self.dist_thres). Use this for city sets that don't share
            KnowAir's roughly-uniform station spacing - a fixed threshold
            either over- or under-connects regions of very different
            density (e.g. it would leave outlying stations like Honolulu/
            Juneau with zero edges - see data/prepare_us_dataset.py).
        """
        self.dist_thres = 3
        self.alti_thres = 1200
        self.use_altitude = use_altitude
        self.k_neighbors = k_neighbors
        self.city_fp = city_fp

        self.altitude = self._load_altitude() if use_altitude else None
        self.nodes = self._gen_nodes()
        self.node_attr = self._add_node_attr()
        self.node_num = len(self.nodes)
        if self.k_neighbors is not None:
            self.edge_index, self.edge_attr = self._gen_edges_knn()
        else:
            self.edge_index, self.edge_attr = self._gen_edges()
            if self.use_altitude:
                self._update_edges()
        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]

    def _load_altitude(self):
        assert os.path.isfile(altitude_fp)
        altitude = np.load(altitude_fp)
        return altitude

    def _lonlat2xy(self, lon, lat, is_aliti):
        if is_aliti:
            lon_l = 100.0
            lon_r = 128.0
            lat_u = 48.0
            lat_d = 16.0
            res = 0.05
        else:
            lon_l = 103.0
            lon_r = 122.0
            lat_u = 42.0
            lat_d = 28.0
            res = 0.125
        x = np.int64(np.round((lon - lon_l - res / 2) / res))
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))
        return x, y

    def _gen_nodes(self):
        nodes = OrderedDict()
        with open(self.city_fp, 'r') as f:
            for line in f:
                # split from the left for idx, then from the right for
                # lon/lat, so a multi-word city name (e.g. "Baton Rouge",
                # written by data/prepare_us_dataset.py) doesn't throw off
                # a plain split(' ') - the original city.txt's one-word
                # China city names still parse identically either way.
                idx, rest = line.rstrip('\n').split(' ', 1)
                city, lon, lat = rest.rsplit(' ', 2)
                idx = int(idx)
                lon, lat = float(lon), float(lat)
                if self.altitude is not None:
                    x, y = self._lonlat2xy(lon, lat, True)
                    altitude = self.altitude[y, x]
                else:
                    altitude = 0.0
                nodes.update({idx: {'city': city, 'altitude': altitude, 'lon': lon, 'lat': lat}})
        return nodes

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]['altitude']
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr

    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]['city']
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats

    def gen_lines(self):

        lines = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))

        return lines

    def _gen_edges(self):
        coords = []
        lonlat = {}
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        dist = distance.cdist(coords, coords, 'euclidean')
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        adj[dist <= self.dist_thres] = 1
        assert adj.shape == dist.shape
        dist = dist * adj
        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        edge_index, dist = edge_index.numpy(), dist.numpy()

        direc_arr = []
        dist_kilometer = []
        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            dist_km = geodesic(src_location, dest_location).kilometers
            v, u = src_lat - dest_lat, src_lon - dest_lon

            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direc = mpcalc.wind_direction(u, v)._magnitude

            direc_arr.append(direc)
            dist_kilometer.append(dist_km)

        direc_arr = np.stack(direc_arr)
        dist_arr = np.stack(dist_kilometer)
        attr = np.stack([dist_arr, direc_arr], axis=-1)

        return edge_index, attr

    def _update_edges(self):
        edge_index = []
        edge_attr = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
            dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)
            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1,0))
            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dest_y, dest_x]
            if np.sum(altitude_points - altitude_src > self.alti_thres) < 3 and \
               np.sum(altitude_points - altitude_dest > self.alti_thres) < 3:
                edge_index.append(self.edge_index[:,i])
                edge_attr.append(self.edge_attr[i])

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)

    def _gen_edges_knn(self):
        """Symmetric k-nearest-neighbor graph by real geodesic distance
        (km), for city sets whose spacing is too uneven for a single fixed
        distance threshold (self._gen_edges) to make sense - e.g. the 51
        US state capitals, where most of CONUS is a few hundred km apart
        but Honolulu/Juneau are thousands of km from every other node. A
        fixed threshold either connects almost nothing out there or (raised
        enough to reach them) over-connects the dense regions instead.
        Every node gets >= k_neighbors edges; edge i->j is kept if j is
        among i's k nearest neighbors OR i is among j's (OR, not AND, so
        the resulting graph is symmetric even though "k nearest" itself
        isn't - a mutual-nearest-neighbor requirement can drop a node with
        an asymmetric neighborhood, like Honolulu/Juneau, to degree 0,
        which is exactly the failure mode this is meant to avoid).
        edge_attr layout matches _gen_edges: [dist_km, direction_deg].
        """
        n = self.node_num
        idx = list(self.nodes.keys())
        lats = np.array([self.nodes[i]['lat'] for i in idx])
        lons = np.array([self.nodes[i]['lon'] for i in idx])

        dist_km = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                if a != b:
                    dist_km[a, b] = geodesic((lats[a], lons[a]), (lats[b], lons[b])).kilometers

        k = min(self.k_neighbors, n - 1)
        adj = np.zeros((n, n), dtype=bool)
        # argsort each row; column 0 is the node itself (distance 0)
        nearest = np.argsort(dist_km, axis=1)[:, 1:k + 1]
        for a in range(n):
            adj[a, nearest[a]] = True
        adj = adj | adj.T
        np.fill_diagonal(adj, False)

        src, dst = np.nonzero(adj)
        edge_index = np.stack([src, dst], axis=0)

        direc_arr = []
        dist_arr = []
        for e in range(edge_index.shape[1]):
            a, b = idx[edge_index[0, e]], idx[edge_index[1, e]]
            dist_arr.append(dist_km[edge_index[0, e], edge_index[1, e]])
            src_lat, src_lon = self.nodes[a]['lat'], self.nodes[a]['lon']
            dest_lat, dest_lon = self.nodes[b]['lat'], self.nodes[b]['lon']
            v, u = src_lat - dest_lat, src_lon - dest_lon
            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direc_arr.append(mpcalc.wind_direction(u, v)._magnitude)

        attr = np.stack([np.stack(dist_arr), np.stack(direc_arr)], axis=-1)
        return edge_index, attr


if __name__ == '__main__':
    graph = Graph()