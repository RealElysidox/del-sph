use nalgebra;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid2Object {
    pub igrid: usize,
    pub iobj: usize,
}

impl PartialOrd for Grid2Object {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Grid2Object {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.igrid.cmp(&other.igrid)
    }
}

#[derive(Debug)]
pub struct SearchGrid {
    pub grid_to_object: Vec<Grid2Object>,
    pub grid_to_object_index: Vec<usize>,
    pub grid_width: f32,
    pub bounding_box_min: nalgebra::Vector3<f32>,
    pub bounding_box_max: nalgebra::Vector3<f32>,
    pub grid_count_x: usize,
    pub grid_count_y: usize,
    pub grid_count_z: usize,
}

impl SearchGrid {
    pub fn new() -> Self {
        Self {
            grid_to_object: Vec::new(),
            grid_to_object_index: Vec::new(),
            grid_width: 1.0,
            bounding_box_min: nalgebra::Vector3::zeros(),
            bounding_box_max: nalgebra::Vector3::repeat(1.0),
            grid_count_x: 1,
            grid_count_y: 1,
            grid_count_z: 1,
        }
    }

    pub fn initialize(
        &mut self,
        bb_min: nalgebra::Vector3<f32>,
        bb_max: nalgebra::Vector3<f32>,
        grid_width: f32,
        num_objects: usize,
    ) {
        self.grid_width = grid_width;
        self.bounding_box_min = bb_min;
        self.bounding_box_max = bb_max;
        self.grid_count_x = ((bb_max.x - bb_min.x) / grid_width).ceil() as usize;
        self.grid_count_y = ((bb_max.y - bb_min.y) / grid_width).ceil() as usize;
        self.grid_count_z = ((bb_max.z - bb_min.z) / grid_width).ceil() as usize;
        self.grid_to_object
            .resize(num_objects, Grid2Object { igrid: 0, iobj: 0 });
    }

    pub fn grid_index(&self, point: nalgebra::Vector3<f32>) -> nalgebra::Vector3<usize> {
        let ix = ((point.x - self.bounding_box_min.x) / self.grid_width)
            .floor()
            .max(0.0)
            .min(self.grid_count_x as f32 - 1.0) as usize;
        let iy = ((point.y - self.bounding_box_min.y) / self.grid_width)
            .floor()
            .max(0.0)
            .min(self.grid_count_y as f32 - 1.0) as usize;
        let iz = ((point.z - self.bounding_box_min.z) / self.grid_width)
            .floor()
            .max(0.0)
            .min(self.grid_count_z as f32 - 1.0) as usize;
        nalgebra::Vector3::new(ix, iy, iz)
    }

    pub fn get_flat_index(&self, point: nalgebra::Vector3<f32>) -> usize {
        let indices = self.grid_index(point);
        indices.z * self.grid_count_y * self.grid_count_x
            + indices.y * self.grid_count_x
            + indices.x
    }

    pub fn post_process(&mut self, is_initial: bool) {
        if is_initial {
            self.grid_to_object.sort();
        } else {
            self.grid_to_object
                .sort_by(|a, b| a.igrid.cmp(&b.igrid));
        }

        let num_grids = self.grid_count_x * self.grid_count_y * self.grid_count_z;
        self.grid_to_object_index.resize(num_grids + 1, 0);
        let mut cur = 0;
        for ig in 0..num_grids {
            while cur < self.grid_to_object.len() && self.grid_to_object[cur].igrid <= ig {
                cur += 1;
            }
            self.grid_to_object_index[ig + 1] = cur;
        }
        // dbg
        for ig in 0..num_grids {
            for i in self.grid_to_object_index[ig]..self.grid_to_object_index[ig + 1] {
                assert_eq!(self.grid_to_object[i].igrid, ig);
            }
        }
    }

    pub fn get_one_ring_neighbors(&self, point: nalgebra::Vector3<f32>) -> Vec<usize> {
        let indices = self.grid_index(point);
        let mut neighbors = Vec::new();

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let x = (indices.x as isize + dx)
                        .max(0)
                        .min(self.grid_count_x as isize - 1) as usize;
                    let y = (indices.y as isize + dy)
                        .max(0)
                        .min(self.grid_count_y as isize - 1) as usize;
                    let z = (indices.z as isize + dz)
                        .max(0)
                        .min(self.grid_count_z as isize - 1) as usize;

                    let grid_index =
                        z * self.grid_count_y * self.grid_count_x + y * self.grid_count_x + x;

                    for i in self.grid_to_object_index[grid_index]
                        ..self.grid_to_object_index[grid_index + 1]
                    {
                        neighbors.push(self.grid_to_object[i].iobj);
                    }
                }
            }
        }
        neighbors
    }
}
