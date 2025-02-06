'use client';

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload } from 'lucide-react';

export default function PlantDiseaseLanding() {
    const [image, setImage] = useState<File | null>(null); // Chỉ giữ 1 ảnh duy nhất
    const [loading, setLoading] = useState(false);
    const [analysis, setAnalysis] = useState<string | null>(null); // Kết quả từ API

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0]; // Chỉ lấy file đầu tiên
        if (!file) return;

        const validTypes = [
            'image/png',
            'image/jpeg',
            'image/jpg',
            'image/webp',
        ];
        if (!validTypes.includes(file.type)) {
            alert('Invalid file type. Please upload an image.');
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            alert('File size must be under 5MB.');
            return;
        }

        setImage(file);
        setAnalysis(null); // Reset kết quả khi chọn ảnh mới
    };

    const handleSubmit = async () => {
        if (!image) {
            alert('Please upload an image before submitting.');
            return;
        }

        setLoading(true);
        setAnalysis(null);

        const formData = new FormData();
        formData.append('file', image);

        try {
            const response = await fetch('http://localhost:8000/api/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.error) {
                setAnalysis(`Error: ${data.error}`);
            } else {
                setAnalysis(
                    `Detected ${data.disease} in ${data.plant} plant with ${(
                        data.confidence * 100
                    ).toFixed(1)}% confidence.`
                );
            }
        } catch (error) {
            setAnalysis(
                'Error: Failed to connect to the server. Please try again.'
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className='min-h-screen bg-green-100 flex flex-col items-center justify-center p-6'>
            <h1 className='text-4xl font-bold text-green-700 mb-4 text-center'>
                AI Plant Disease Detection
            </h1>
            <p className='text-lg text-gray-600 mb-6 text-center max-w-lg'>
                Upload an image of a plant leaf to detect potential diseases
                using our AI-powered analysis.
            </p>

            <Card className='p-6 w-full max-w-lg bg-white shadow-xl rounded-2xl text-center'>
                <label className='cursor-pointer border-dashed border-2 border-green-500 p-6 rounded-xl flex flex-col items-center justify-center'>
                    <Upload className='w-12 h-12 text-green-500 mb-3' />
                    <span className='text-gray-600'>
                        Click to upload (Max: 1 image, 5MB)
                    </span>
                    <input
                        type='file'
                        accept='image/png, image/jpeg, image/jpg, image/webp'
                        className='hidden'
                        onChange={handleFileChange}
                    />
                </label>
            </Card>

            {image && (
                <div className='mt-6'>
                    <h2 className='text-lg font-semibold text-gray-700'>
                        Preview:
                    </h2>
                    <img
                        src={URL.createObjectURL(image)}
                        alt='Uploaded preview'
                        className='w-64 h-64 object-cover rounded-xl shadow-md mt-2'
                    />
                </div>
            )}

            {image && (
                <Button
                    onClick={handleSubmit}
                    className='mt-6 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg'
                >
                    {loading ? 'Analyzing...' : 'Submit for Analysis'}
                </Button>
            )}

            {analysis && (
                <div className='mt-6 bg-white p-4 rounded-lg shadow-md w-full max-w-lg'>
                    <h2 className='text-lg font-semibold text-gray-700'>
                        AI Diagnosis:
                    </h2>
                    <p className='text-gray-600 mt-2'>{analysis}</p>
                </div>
            )}
        </div>
    );
}
