import pytest
from unittest.mock import Mock, patch, MagicMock
import cv2
import numpy as np
from src.infrastructure.image.image_sharpness_analyzer import ImageSharpnessAnalyzer  # Замените на ваш модуль


class TestImageSharpnessAnalyzer:
    """Тесты для ImageSharpnessAnalyzer"""

    # Тесты для calculate_face_sharpness
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    @patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale')
    @patch('cv2.Laplacian')
    def test_calculate_face_sharpness_nonexistent_path(self, mock_laplacian, mock_detect, mock_cvt, mock_imread):
        mock_imread.return_value = None
        result = ImageSharpnessAnalyzer.calculate_face_sharpness('nonexistent.jpg')
        assert result == 0.0

    @patch('cv2.imread')
    def test_calculate_face_sharpness_invalid_format(self, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'empty', return_value=True):
            result = ImageSharpnessAnalyzer.calculate_face_sharpness('invalid.txt')
            assert result == 0.0

    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_calculate_face_sharpness_no_faces(self, mock_cvt, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvt.return_value = np.zeros((100, 100), dtype=np.uint8)
        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale', return_value=[]):
            result = ImageSharpnessAnalyzer.calculate_face_sharpness('no_faces.jpg')
            assert result == 0.0

    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_calculate_face_sharpness_single_face(self, mock_cvt, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvt.return_value = np.zeros((100, 100), dtype=np.uint8)

        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale', return_value=[(10, 10, 50, 50)]):
            mock_laplacian = Mock()
            mock_laplacian.var.return_value = 100.0
            with patch('cv2.Laplacian', return_value=mock_laplacian):
                result = ImageSharpnessAnalyzer.calculate_face_sharpness('single_face.jpg')
                assert result > 0

    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_calculate_face_sharpness_multiple_faces(self, mock_cvt, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvt.return_value = np.zeros((100, 100), dtype=np.uint8)

        # Два лица: первое большее (60x60), второе меньшее (30x30)
        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale',
                          return_value=[(10, 10, 60, 60), (80, 80, 30, 30)]):
            mock_laplacian = Mock()
            mock_laplacian.var.return_value = 150.0
            with patch('cv2.Laplacian', return_value=mock_laplacian):
                result = ImageSharpnessAnalyzer.calculate_face_sharpness('multiple_faces.jpg')
                # Должен использоваться первый (больший) face ROI
                assert result == 150.0

    @patch('cv2.imread')
    def test_calculate_face_sharpness_cascade_not_loaded(self, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'empty', return_value=True):
            result = ImageSharpnessAnalyzer.calculate_face_sharpness('any.jpg')
            assert result == 0.0

    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_sharpness_comparison(self, mock_cvt, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvt.return_value = np.zeros((100, 100), dtype=np.uint8)

        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale', return_value=[(10, 10, 50, 50)]):
            # Резкое изображение
            sharp_laplacian = Mock()
            sharp_laplacian.var.return_value = 500.0

            # Размытое изображение
            blurry_laplacian = Mock()
            blurry_laplacian.var.return_value = 50.0

            with patch('cv2.Laplacian') as mock_laplacian:
                mock_laplacian.side_effect = [sharp_laplacian, blurry_laplacian]
                sharp_result = ImageSharpnessAnalyzer.calculate_face_sharpness('sharp.jpg')
                blurry_result = ImageSharpnessAnalyzer.calculate_face_sharpness('blurry.jpg')

                assert sharp_result > blurry_result

    # Тесты для get_image_info
    @patch('cv2.imread')
    def test_get_image_info_nonexistent_file(self, mock_imread):
        mock_imread.return_value = None
        result = ImageSharpnessAnalyzer.get_image_info('nonexistent.jpg')
        assert result == (0, 0, 0, False)

    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_get_image_info_no_faces(self, mock_cvt, mock_imread):
        mock_imread.return_value = np.zeros((100, 50, 3), dtype=np.uint8)
        mock_cvt.return_value = np.zeros((100, 50), dtype=np.uint8)

        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale', return_value=[]):
            width, height, area, has_face = ImageSharpnessAnalyzer.get_image_info('no_faces.jpg')
            assert width == 50
            assert height == 100
            assert area == 5000
            assert not has_face

    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_get_image_info_with_faces(self, mock_cvt, mock_imread):
        mock_imread.return_value = np.zeros((200, 100, 3), dtype=np.uint8)
        mock_cvt.return_value = np.zeros((200, 100), dtype=np.uint8)

        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale', return_value=[(0, 0, 10, 10)]):
            width, height, area, has_face = ImageSharpnessAnalyzer.get_image_info('with_faces.jpg')
            assert width == 100
            assert height == 200
            assert area == 20000
            assert has_face

    @patch('cv2.imread')
    def test_get_image_info_area_calculation(self, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale', return_value=[]):
            width, height, area, has_face = ImageSharpnessAnalyzer.get_image_info('100x100.jpg')
            assert area == 10000

    @patch('cv2.imread')
    def test_get_image_info_minimal_size(self, mock_imread):
        mock_imread.return_value = np.zeros((1, 1, 3), dtype=np.uint8)
        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale', return_value=[]):
            width, height, area, has_face = ImageSharpnessAnalyzer.get_image_info('1x1.jpg')
            assert width == 1
            assert height == 1
            assert area == 1

    # Интеграционные тесты
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_methods_consistency(self, mock_cvt, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvt.return_value = np.zeros((100, 100), dtype=np.uint8)

        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale', return_value=[(10, 10, 50, 50)]):
            # Сначала проверяем наличие лица
            width, height, area, has_face = ImageSharpnessAnalyzer.get_image_info('test.jpg')
            assert has_face

            # Затем проверяем, что резкость вычисляется
            mock_laplacian = Mock()
            mock_laplacian.var.return_value = 100.0
            with patch('cv2.Laplacian', return_value=mock_laplacian):
                sharpness = ImageSharpnessAnalyzer.calculate_face_sharpness('test.jpg')
                assert sharpness > 0

    # Тестирование типов данных
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    def test_return_types(self, mock_cvt, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvt.return_value = np.zeros((100, 100), dtype=np.uint8)

        # Тест calculate_face_sharpness
        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'detectMultiScale', return_value=[]):
            sharpness = ImageSharpnessAnalyzer.calculate_face_sharpness('test.jpg')
            assert isinstance(sharpness, float)

        # Тест get_image_info
        info = ImageSharpnessAnalyzer.get_image_info('test.jpg')
        assert isinstance(info, tuple)
        assert len(info) == 4
        assert all(isinstance(i, (int, bool)) for i in info)

    # Тестирование обработки ошибок
    def test_none_path(self):
        with pytest.raises(Exception):
            ImageSharpnessAnalyzer.calculate_face_sharpness(None)  # type: ignore

    # Тестирование логирования (используем capsys для перехвата вывода)
    @patch('cv2.imread')
    def test_error_logging(self, mock_imread, capsys):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(ImageSharpnessAnalyzer.face_cascade, 'empty', return_value=True):
            ImageSharpnessAnalyzer.calculate_face_sharpness('test.jpg')
            captured = capsys.readouterr()
            assert "Не удалось загрузить каскад" in captured.out


if __name__ == '__main__':
    pytest.main()