import asyncio
import math
import sys
from pathlib import Path

import numpy as np
import pygame
from pygame.locals import K_BACKSPACE, K_DELETE, K_ESCAPE, KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, QUIT

import Three_States as dm


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "figures"
TRACTS_DIR = PROJECT_ROOT / "tracts"

ICONSIZE = 30
FPS = 20
WINDOWWIDTH = int(1.4 * 917)
WINDOWHEIGHT = int(1.4 * 582)

BRIGHTBLUE = (0, 50, 255)
RED = (255, 64, 64)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

LEFT = 1
RIGHT = 3

MED = "med"
FIRE = "fire"
JOINT = "joint"

UPLEFT = (44.988032, -93.207538)
UPLEFT_PIX = (75, 43)
DOWNRIGHT = (44.890738, -93.004278)
DOWNRIGHT_PIX = (1140, 768)

TRACTS_PIX = [
    (52, 27),
    (283, 9),
    (386, 5),
    (498, 7),
    (607, 11),
    (700, 5),
    (925, 7),
    (505, 83),
    (702, 86),
    (768, 71),
    (820, 60),
    (390, 130),
    (502, 160),
    (610, 117),
    (719, 165),
    (820, 141),
    (851, 118),
    (987, 100),
    (78, 101),
    (208, 183),
    (259, 171),
    (306, 179),
    (341, 183),
    (396, 195),
    (395, 245),
    (503, 207),
    (555, 210),
    (688, 174),
    (762, 207),
    (76, 193),
    (157, 278),
    (243, 281),
    (395, 281),
    (502, 279),
    (555, 281),
    (395, 320),
    (480, 314),
    (529, 322),
    (620, 289),
    (729, 262),
    (808, 239),
    (849, 231),
    (990, 226),
    (153, 346),
    (205, 345),
    (207, 389),
    (287, 350),
    (342, 337),
    (489, 350),
    (394, 406),
    (491, 352),
    (558, 351),
    (544, 370),
    (715, 348),
    (179, 445),
    (234, 443),
    (288, 444),
    (341, 443),
    (354, 440),
    (444, 443),
    (490, 405),
    (493, 457),
    (621, 373),
    (691, 478),
    (960, 327),
    (181, 551),
    (109, 570),
    (605, 224),
    (396, 350),
    (73, 258),
    (840, 349),
]

CENSUS_TRACTS = [
    (44.97985, -93.1929),
    (44.97791, -93.15685),
    (44.98301, -93.13659),
    (44.98617, -93.11737),
    (44.98301, -93.0978),
    (44.98714, -93.06999),
    (44.98471, -93.02523),
    (44.97524, -93.11702),
    (44.97451, -93.08303),
    (44.97597, -93.07205),
    (44.97912, -93.05522),
    (44.97062, -93.13556),
    (44.96892, -93.11599),
    (44.96941, -93.0978),
    (44.96771, -93.07754),
    (44.97184, -93.05625),
    (44.96917, -93.04149),
    (44.97029, -93.02022),
    (44.96771, -93.19221),
    (44.96286, -93.17722),
    (44.96406, -93.16915),
    (44.96115, -93.16228),
    (44.96285, -93.15199),
    (44.96334, -93.13728),
    (44.95823, -93.13693),
    (44.96091, -93.12114),
    (44.96018, -93.11153),
    (44.95993, -93.08372),
    (44.96188, -93.06896),
    (44.95872, -93.19255),
    (44.95022, -93.18431),
    (44.95143, -93.16097),
    (44.95362, -93.13728),
    (44.95337, -93.12114),
    (44.95337, -93.11153),
    (44.94876, -93.13934),
    (44.94876, -93.12629),
    (44.94827, -93.11325),
    (44.94852, -93.09196),
    (44.95167, -93.07205),
    (44.95532, -93.06003),
    (44.95409, -93.04558),
    (44.95676, -93.02021),
    (44.94098, -93.18775),
    (44.9439, -93.17539),
    (44.93782, -93.17539),
    (44.94001, -93.16234),
    (44.94098, -93.15273),
    (44.94603, -93.12115),
    (44.93636, -93.13693),
    (44.94074, -93.12286),
    (44.94258, -93.11019),
    (44.93532, -93.11242),
    (44.93515, -93.06827),
    (44.92616, -93.18328),
    (44.92713, -93.17264),
    (44.92689, -93.16268),
    (44.92397, -93.15204),
    (44.91838, -93.14071),
    (44.92591, -93.13282),
    (44.93053, -93.1232),
    (44.92421, -93.1081),
    (44.93564, -93.09333),
    (44.9247, -93.07857),
    (44.93418, -93.01574),
    (44.91254, -93.17024),
    (44.89782, -93.17799),
    (44.95799, -93.09883),
    (44.94293, -93.13797),
    (44.93831, -93.19733),
    (44.91862, -93.03428),
]

SAINT_PAUL_STATIONS = [
    (44.95099318068965, -93.08500943661971),
    (44.98105367172413, -93.03405129577465),
    (44.97555152827586, -93.18043666666667),
    (44.94884600275862, -93.13195963380282),
    (44.97179396689655, -93.0447391455399),
    (44.95099318068965, -93.08500943661971),
    (44.98091947310345, -93.03386044131456),
    (44.97555152827586, -93.18043666666667),
    (44.93005819586207, -93.13863953990611),
    (44.965620830344825, -93.08577285446009),
    (44.92911880551724, -93.09207105164319),
    (44.95220096827586, -93.16650429107982),
    (44.98293245241379, -93.06172519248827),
    (44.962936857931034, -93.13195963380282),
    (44.91717512827586, -93.1760470140845),
    (44.97420954206896, -93.10905709859155),
    (44.95179837241379, -93.02794395305165),
]

EMS_FRACTIONS = np.array(
    [
        0.01158147,
        0.02600284,
        0.00940717,
        0.0192137,
        0.01717252,
        0.01606319,
        0.02067803,
        0.01757188,
        0.01091587,
        0.01672879,
        0.00603479,
        0.00891906,
        0.00993965,
        0.0115371,
        0.02116613,
        0.01078275,
        0.03572062,
        0.01792687,
        0.01797125,
        0.00559105,
        0.01388889,
        0.00301739,
        0.00563543,
        0.00745474,
        0.01610756,
        0.01238019,
        0.00954029,
        0.03310259,
        0.00554668,
        0.01766063,
        0.00261803,
        0.02671282,
        0.01690628,
        0.00474796,
        0.02334043,
        0.00519169,
        0.00505857,
        0.01393326,
        0.03421193,
        0.00803159,
        0.01167022,
        0.03625311,
        0.03518814,
        0.00532481,
        0.00306177,
        0.0041711,
        0.00430422,
        0.00430422,
        0.0056798,
        0.00230742,
        0.00736599,
        0.01215832,
        0.00736599,
        0.0063454,
        0.00789847,
        0.00616791,
        0.00412673,
        0.01042776,
        0.02236422,
        0.00940717,
        0.00954029,
        0.00430422,
        0.03430067,
        0.00745474,
        0.02538161,
        0.01264643,
        0.03421193,
        0.06948882,
        0.00794285,
        0.01206958,
        0.00075435,
    ]
)

FIRE_FRACTIONS = np.array(
    [
        0.0144893,
        0.02629539,
        0.00888438,
        0.01252161,
        0.01222348,
        0.01329676,
        0.03631268,
        0.01103095,
        0.00840737,
        0.01532407,
        0.00924214,
        0.00590305,
        0.01234273,
        0.00775148,
        0.00727446,
        0.01043468,
        0.04239461,
        0.02593763,
        0.02522211,
        0.00673782,
        0.01168684,
        0.01109057,
        0.00453163,
        0.00816886,
        0.01043468,
        0.00787073,
        0.00643969,
        0.01454892,
        0.00888438,
        0.0267724,
        0.00465088,
        0.02426808,
        0.01639735,
        0.00590305,
        0.01341602,
        0.00608193,
        0.00787073,
        0.01699362,
        0.094568,
        0.00930177,
        0.01109057,
        0.02581838,
        0.02504323,
        0.00697633,
        0.0035776,
        0.0051279,
        0.00524715,
        0.00572417,
        0.00918252,
        0.00381611,
        0.0111502,
        0.01454892,
        0.00798998,
        0.00942102,
        0.00477014,
        0.00381611,
        0.0051279,
        0.00679745,
        0.01603959,
        0.00709558,
        0.06159442,
        0.00548566,
        0.02098861,
        0.00965953,
        0.0247451,
        0.00954028,
        0.02045197,
        0.01168684,
        0.01753026,
        0.00673782,
        0.00131179,
    ]
)


def build_color_gradient(start_color, end_color, steps):
    start = np.array(start_color, dtype=float)
    end = np.array(end_color, dtype=float)
    return [
        tuple(int(round(value)) for value in color)
        for color in np.linspace(start, end, steps)
    ]


COLORS = build_color_gradient((244, 32, 8), (156, 245, 66), 20)


def fill_surface_rgb(surface, color):
    pixels = pygame.surfarray.pixels3d(surface)
    pixels[:, :] = color
    del pixels


def census_tracts():
    return list(CENSUS_TRACTS)


def saint_paul_stations():
    return list(SAINT_PAUL_STATIONS)


def pixel_to_coordinates(pos):
    lat = UPLEFT[0] + (pos[1] - UPLEFT_PIX[1]) * (DOWNRIGHT[0] - UPLEFT[0]) / (DOWNRIGHT_PIX[1] - UPLEFT_PIX[1])
    lon = UPLEFT[1] + (pos[0] - UPLEFT_PIX[0]) * (DOWNRIGHT[1] - UPLEFT[1]) / (DOWNRIGHT_PIX[0] - UPLEFT_PIX[0])
    return (lat, lon)


def coordinates_to_pixel(latlong):
    pos_y = UPLEFT_PIX[1] + (latlong[0] - UPLEFT[0]) * (DOWNRIGHT_PIX[1] - UPLEFT_PIX[1]) / (DOWNRIGHT[0] - UPLEFT[0])
    pos_x = UPLEFT_PIX[0] + (latlong[1] - UPLEFT[1]) * (DOWNRIGHT_PIX[0] - UPLEFT_PIX[0]) / (DOWNRIGHT[1] - UPLEFT[1])
    return (int(round(pos_x)), int(round(pos_y)))


def haversine_distance_m(point_a, point_b):
    lat1, lon1 = map(math.radians, point_a)
    lat2, lon2 = map(math.radians, point_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)
    hav = sin_dlat * sin_dlat + math.cos(lat1) * math.cos(lat2) * sin_dlon * sin_dlon
    return 2.0 * 6371008.8 * math.asin(min(1.0, math.sqrt(hav)))


def get_distance(latlong_list):
    tract_points = census_tracts()
    return np.array(
        [
            [haversine_distance_m(tract_point, unit_point) for unit_point in latlong_list]
            for tract_point in tract_points
        ],
        dtype=float,
    )


pix_to_corrdinates = pixel_to_coordinates
corrdinates_to_pix = coordinates_to_pixel
fill = fill_surface_rgb


class UnitSprite(pygame.sprite.Sprite):
    def __init__(self, kind, image, pos):
        super().__init__()
        self.kind = kind
        self.image = image
        self.rect = self.image.get_rect(center=pos)
        self.pos = pos

    def update(self, pos):
        self.rect = self.image.get_rect(center=pos)
        self.pos = pos


class MapHypercubeGame:
    def __init__(self):
        self.running = True
        self.initialized = False

        self.clock = None
        self.display = None
        self.font = None
        self.font_medium = None
        self.font_large = None

        self.base_map = None
        self.bg_tract = None
        self.tract_images = []
        self.token_images = {}
        self.button_images = {}

        self.med_group = pygame.sprite.Group()
        self.fire_group = pygame.sprite.Group()
        self.joint_group = pygame.sprite.Group()

        self.census_latlong = census_tracts()
        self.selected_token = None

        self.drag_kind = None
        self.drag_pos = None
        self.dragging_sprite = None

        self.indicator = 0
        self.uti_indicator = 0

        self.time_e = np.array([])
        self.time_f = np.array([])
        self.mean_e = None
        self.mean_f = None
        self.rho_e = np.array([])
        self.rho_f = np.array([])
        self.color_map_e = np.array([], dtype=int)
        self.color_map_f = np.array([], dtype=int)

    def initialize(self):
        if self.initialized:
            return

        pygame.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        pygame.display.set_caption("Saint Paul, MN")

        self.font = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 42)

        self.cal_rect = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), 0, ICONSIZE, ICONSIZE)
        self.med_pile_rect = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(6 * ICONSIZE / 2), ICONSIZE, ICONSIZE)
        self.fire_pile_rect = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(3 * ICONSIZE / 2), ICONSIZE, ICONSIZE)
        self.joint_pile_rect = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(9 * ICONSIZE / 2), ICONSIZE, ICONSIZE)
        self.ori_button_rect = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(12 * ICONSIZE / 2), ICONSIZE, ICONSIZE)
        self.clear_button_rect = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(15 * ICONSIZE / 2), ICONSIZE, ICONSIZE)
        self.indi_button_rect = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(18 * ICONSIZE / 2), ICONSIZE, ICONSIZE)
        self.uti_button_rect = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(21 * ICONSIZE / 2), ICONSIZE, ICONSIZE)

        self.load_assets()
        self.initialized = True

    def load_assets(self):
        self.base_map = self.load_image(FIGURES_DIR / "map.png", size=(WINDOWWIDTH, WINDOWHEIGHT))
        self.bg_tract = self.load_image(FIGURES_DIR / "tracts.png", size=(1073, 792))

        self.token_images = {
            MED: self.load_image(FIGURES_DIR / "med.png", size=(ICONSIZE, ICONSIZE)),
            FIRE: self.load_image(FIGURES_DIR / "fire.png", size=(ICONSIZE, ICONSIZE)),
            JOINT: self.load_image(FIGURES_DIR / "joint.png", size=(ICONSIZE, ICONSIZE)),
        }
        self.button_images = {
            "calculate": self.load_image(FIGURES_DIR / "start.png", size=(ICONSIZE, ICONSIZE)),
            "original": self.load_image(FIGURES_DIR / "stpaul.jpg", size=(ICONSIZE, ICONSIZE)),
            "clear": self.load_image(FIGURES_DIR / "clear-icon.png", size=(ICONSIZE, ICONSIZE)),
            "indicator": self.load_image(FIGURES_DIR / "switch.png", size=(ICONSIZE, ICONSIZE)),
            "utilization": self.load_image(FIGURES_DIR / "uti.png", size=(ICONSIZE, ICONSIZE)),
        }

        tract_files = sorted(TRACTS_DIR.glob("*.png"), key=lambda path: int(path.stem))
        self.tract_images = [self.load_image(path, scale=0.58) for path in tract_files]

        if len(self.tract_images) != len(TRACTS_PIX):
            raise ValueError("The number of tract images does not match the hard-coded tract positions.")

    def load_image(self, path, size=None, scale=None):
        image = pygame.image.load(str(path)).convert_alpha()
        if size is not None:
            return pygame.transform.smoothscale(image, size)
        if scale is not None:
            return pygame.transform.rotozoom(image, 0, scale)
        return image

    def clear_analysis(self):
        self.time_e = np.array([])
        self.time_f = np.array([])
        self.mean_e = None
        self.mean_f = None
        self.rho_e = np.array([])
        self.rho_f = np.array([])
        self.color_map_e = np.array([], dtype=int)
        self.color_map_f = np.array([], dtype=int)

    def can_analyze(self):
        return (len(self.med_group) + len(self.joint_group)) >= 2 and (len(self.fire_group) + len(self.joint_group)) >= 2

    def group_for_kind(self, kind):
        if kind == MED:
            return self.med_group
        if kind == FIRE:
            return self.fire_group
        return self.joint_group

    def make_unit(self, kind, pos):
        return UnitSprite(kind, self.token_images[kind], pos)

    def sprites_for_hit_test(self):
        return list(self.joint_group) + list(self.fire_group) + list(self.med_group)

    def find_sprite(self, pos):
        for sprite in self.sprites_for_hit_test():
            if sprite.rect.collidepoint(pos):
                return sprite
        return None

    def delete_token_at(self, pos):
        sprite = self.find_sprite(pos)
        if sprite is None:
            return False

        if self.selected_token is sprite:
            self.selected_token = None
        sprite.kill()
        self.maybe_analyze()
        return True

    def maybe_analyze(self):
        if self.can_analyze():
            self.analyze()
        else:
            self.clear_analysis()

    def add_original_stations(self):
        for index, latlong in enumerate(saint_paul_stations()):
            pix = coordinates_to_pixel(latlong)
            if index < 3:
                self.med_group.add(self.make_unit(MED, (pix[0] - 20, pix[1])))
            elif index < 8:
                self.fire_group.add(self.make_unit(FIRE, pix))
            else:
                self.joint_group.add(self.make_unit(JOINT, pix))

    def start_drag_from_pile(self, kind, pos):
        self.drag_kind = kind
        self.drag_pos = pos
        self.dragging_sprite = None

    def start_drag_existing(self, sprite, pos):
        self.selected_token = sprite
        self.drag_kind = sprite.kind
        self.drag_pos = pos
        self.dragging_sprite = sprite
        sprite.update(pos)

    def stop_drag(self, pos):
        if self.dragging_sprite is None:
            self.group_for_kind(self.drag_kind).add(self.make_unit(self.drag_kind, pos))
        else:
            self.dragging_sprite.update(pos)

        self.drag_kind = None
        self.drag_pos = None
        self.dragging_sprite = None
        self.maybe_analyze()

    def cancel_drag(self):
        self.drag_kind = None
        self.drag_pos = None
        self.dragging_sprite = None

    def handle_keydown(self, event):
        if event.key in (K_DELETE, K_BACKSPACE) and self.selected_token is not None:
            self.selected_token.kill()
            self.selected_token = None
            self.maybe_analyze()
        elif event.key == K_ESCAPE:
            self.cancel_drag()
            self.selected_token = None

    def handle_mouse_down(self, event):
        if event.button == RIGHT:
            if self.delete_token_at(event.pos):
                return
            if self.indi_button_rect.collidepoint(event.pos):
                self.indicator = 2
            return

        if event.button != LEFT or self.drag_kind is not None:
            return

        if self.cal_rect.collidepoint(event.pos):
            if self.can_analyze():
                self.analyze()
            return

        if self.ori_button_rect.collidepoint(event.pos):
            self.add_original_stations()
            self.maybe_analyze()
            return

        if self.clear_button_rect.collidepoint(event.pos):
            self.med_group.empty()
            self.fire_group.empty()
            self.joint_group.empty()
            self.selected_token = None
            self.clear_analysis()
            return

        if self.indi_button_rect.collidepoint(event.pos):
            self.indicator = (self.indicator + 1) % 2
            self.apply_indicator_colors()
            return

        if self.uti_button_rect.collidepoint(event.pos):
            self.uti_indicator = (self.uti_indicator + 1) % 2
            return

        sprite = self.find_sprite(event.pos)
        if sprite is not None:
            self.start_drag_existing(sprite, event.pos)
            return

        if self.med_pile_rect.collidepoint(event.pos):
            self.start_drag_from_pile(MED, event.pos)
            return

        if self.fire_pile_rect.collidepoint(event.pos):
            self.start_drag_from_pile(FIRE, event.pos)
            return

        if self.joint_pile_rect.collidepoint(event.pos):
            self.start_drag_from_pile(JOINT, event.pos)
            return

        self.selected_token = None

    def handle_mouse_motion(self, event):
        if self.drag_kind is None:
            return

        self.drag_pos = event.pos
        if self.dragging_sprite is not None:
            self.dragging_sprite.update(event.pos)

    def handle_mouse_up(self, event):
        if event.button != LEFT or self.drag_kind is None:
            return
        self.stop_drag(event.pos)

    def handle_events(self, events):
        for event in events:
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                self.handle_keydown(event)
            elif event.type == MOUSEBUTTONDOWN:
                self.handle_mouse_down(event)
            elif event.type == MOUSEMOTION:
                self.handle_mouse_motion(event)
            elif event.type == MOUSEBUTTONUP:
                self.handle_mouse_up(event)

    def apply_indicator_colors(self):
        if self.time_e.size == 0 or self.indicator not in (0, 1):
            return

        color_map = self.color_map_e if self.indicator == 0 else self.color_map_f
        for index, tract_image in enumerate(self.tract_images):
            fill_surface_rgb(tract_image, COLORS[19 - int(color_map[index])])

    def analyze(self):
        latlong_list = [pixel_to_coordinates(med.pos) for med in self.med_group]
        latlong_list += [pixel_to_coordinates(fire.pos) for fire in self.fire_group]
        latlong_list += [pixel_to_coordinates(joint.pos) for joint in self.joint_group]

        n_1 = len(self.med_group)
        n_2 = len(self.fire_group)
        n_d = len(self.joint_group)
        n_units = n_1 + n_2 + n_d

        if n_units == 0:
            self.clear_analysis()
            return

        distance_matrix = get_distance(latlong_list) * 1.37
        distance_matrix = distance_matrix / 1600.0

        mask_long = distance_matrix > 0.7712735431536174
        distance_matrix[mask_long] = (distance_matrix[mask_long] * 111.51113889304331 + 86.005591195132666) / 60.0
        distance_matrix[~mask_long] = 195.86302790816589 * np.sqrt(distance_matrix[~mask_long]) / 60.0

        units_e = np.array([idx for idx in range(n_units) if idx not in range(n_1, n_1 + n_2)])
        units_f = np.array([idx for idx in range(n_units) if idx not in range(n_1)])

        t_mat_1 = distance_matrix[:, units_e]
        t_mat_2 = distance_matrix[:, units_f]
        pre_list_1 = t_mat_1.argsort(axis=1)
        pre_list_2 = t_mat_2.argsort(axis=1)

        hypercube = dm.Three_State_Hypercube(
            {
                "Lambda_1": 57.80867911518289,
                "Mu_1": 42.17,
                "Lambda_2": 40.266999999999996,
                "Mu_2": 51.3,
            }
        )
        hypercube.Update_Parameters(
            N=n_units,
            N_1=n_1,
            N_2=n_2,
            K=71,
            pre_list_1=pre_list_1,
            pre_list_2=pre_list_2,
            frac_j_1=EMS_FRACTIONS,
            frac_j_2=FIRE_FRACTIONS,
            t_mat_1=t_mat_1,
            t_mat_2=t_mat_2,
        )
        hypercube.Creat_Two_Subsystems()
        hypercube.Linear_Alpha()

        self.rho_e = hypercube.sub1.rho_approx
        self.rho_f = hypercube.sub2.rho_approx
        mrt_1, mrt_2, mrt_1_j, mrt_2_j = hypercube.Get_MRT_Approx_3state()

        self.time_e = np.asarray(mrt_1_j + 1.75)
        self.time_f = np.asarray(mrt_2_j + 1.67 - 0.04)
        self.mean_e = float(mrt_1 + 1.75)
        self.mean_f = float(mrt_2 + 1.67 - 0.04)

        self.color_map_e = np.clip(((self.time_e - 4) * 20 / 6).astype(int), 0, 19)
        self.color_map_f = np.clip(((self.time_f - 4) * 20 / 6).astype(int), 0, 19)
        self.apply_indicator_colors()

    def draw_response_overlay(self):
        if self.time_e.size == 0:
            return

        if self.indicator in (0, 1):
            values = self.time_e if self.indicator == 0 else self.time_f
            for tract_image, tract_pos in zip(self.tract_images, TRACTS_PIX):
                self.display.blit(tract_image, tract_pos)

            self.display.blit(self.bg_tract, (70, 7))
            for value, tract_point in zip(values, self.census_latlong):
                label = self.font.render(f"{value:.2f}", True, BLACK)
                self.display.blit(label, coordinates_to_pixel(tract_point))

        if self.indicator == 0:
            ems_font = self.font_large
            fire_font = self.font_medium
        elif self.indicator == 1:
            ems_font = self.font_medium
            fire_font = self.font_large
        else:
            ems_font = self.font_medium
            fire_font = self.font_medium

        self.display.blit(ems_font.render(f"EMS: {self.mean_e:.2f} min", True, RED), (620, 630))
        self.display.blit(fire_font.render(f"Fire: {self.mean_f:.2f} min", True, BRIGHTBLUE), (620, 670))

    def draw_utilization_overlay(self):
        if self.uti_indicator != 1 or self.rho_e.size == 0 or self.rho_f.size == 0:
            return

        ems_index = 0
        fire_index = 0

        for med in self.med_group:
            text = self.font_medium.render(f"{self.rho_e[ems_index] * 100:.2f}%", True, RED)
            self.display.blit(text, (med.pos[0] - 30, med.pos[1] - 35))
            ems_index += 1

        for fire in self.fire_group:
            text = self.font_medium.render(f"{self.rho_f[fire_index] * 100:.2f}%", True, BRIGHTBLUE)
            self.display.blit(text, (fire.pos[0] - 25, fire.pos[1] + 15))
            fire_index += 1

        for joint in self.joint_group:
            text_e = self.font_medium.render(f"{self.rho_e[ems_index] * 100:.2f}%", True, RED)
            text_f = self.font_medium.render(f"{self.rho_f[fire_index] * 100:.2f}%", True, BRIGHTBLUE)
            self.display.blit(text_e, (joint.pos[0] - 25, joint.pos[1] - 35))
            self.display.blit(text_f, (joint.pos[0] - 25, joint.pos[1] + 15))
            ems_index += 1
            fire_index += 1

    def draw_tokens(self):
        self.med_group.draw(self.display)
        self.fire_group.draw(self.display)
        self.joint_group.draw(self.display)

        if self.selected_token is not None and self.selected_token.alive():
            pygame.draw.rect(self.display, BLACK, self.selected_token.rect.inflate(6, 6), 2)

        if self.drag_kind is not None and self.dragging_sprite is None and self.drag_pos is not None:
            self.display.blit(
                self.token_images[self.drag_kind],
                (self.drag_pos[0] - ICONSIZE // 2, self.drag_pos[1] - ICONSIZE // 2, ICONSIZE, ICONSIZE),
            )

    def draw_controls(self):
        self.display.blit(self.token_images[MED], self.med_pile_rect)
        self.display.blit(self.token_images[FIRE], self.fire_pile_rect)
        self.display.blit(self.token_images[JOINT], self.joint_pile_rect)
        self.display.blit(self.button_images["calculate"], self.cal_rect)
        self.display.blit(self.button_images["original"], self.ori_button_rect)
        self.display.blit(self.button_images["clear"], self.clear_button_rect)
        self.display.blit(self.button_images["indicator"], self.indi_button_rect)
        self.display.blit(self.button_images["utilization"], self.uti_button_rect)

        if sys.platform == "emscripten":
            hint = self.font.render("Select a unit, then press Delete or Backspace to remove it.", True, BLACK)
            self.display.blit(hint, (12, WINDOWHEIGHT - 20))

    def draw(self):
        self.display.blit(self.base_map, (0, 0))
        self.draw_response_overlay()
        self.draw_utilization_overlay()
        self.draw_tokens()
        self.draw_controls()

    def step(self):
        self.handle_events(pygame.event.get())
        self.draw()
        pygame.display.update()

    def shutdown(self):
        pygame.quit()

    def run(self):
        self.initialize()
        while self.running:
            self.step()
            self.clock.tick(FPS)
        self.shutdown()

    async def run_async(self):
        self.initialize()
        while self.running:
            self.step()
            if sys.platform != "emscripten":
                self.clock.tick(FPS)
            await asyncio.sleep(0)
        self.shutdown()


def main():
    game = MapHypercubeGame()
    game.run()


async def main_async():
    game = MapHypercubeGame()
    await game.run_async()


if __name__ == "__main__":
    main()
